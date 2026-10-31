# 검증 가능한 오답노트형 자기수정 RL: 연구·아키텍처·구현 계약

작성 기준: 2026-10-26 KST. 대상: `C:\projects\multimodel\fine_tuning`.

설계 기준 상태: **DESIGNED (2026-10-26)**. 이후 source10의 실제 최소 기능 검증 F를 통과했고 같은 설정의 고정10을 시작했다(2026-10-28). 과거 동결8 고정10의 시간한도 종료와 함께 보존한다. 정확한 전체노트 학습·수정 의미 S·효과 E는 미완료다. 최초 설계·이후 구현·각 실행 근거를 구분한다. 모델 다운로드·RunPod 실행·GitHub 게시를 추가하지 않았다.

같이 읽을 파일: [단계별 구현 계획](plan.md), [오프라인 전체 구조도](architecture.html).

### 2026-10-28 구현·실행 상태 갱신 — source10

**최신 종료 근거:** 최소2문항3509.993초/exit0·실제update1·L252/M24/V48 변경과624실물텐서 검사를 완료했다. 별도PID의 fresh reload도 exit0이며 optimizer/step1·단일/복수 이미지 추론/생성 일치·원본접근0·NF4불변이다. 53연결파일/392동결소스를 주재해시했다. `training-reload-closure-main-v10.json` SHA `c30bfa2d…`가 정본이다. F=true/S=false/E=false이고 online/final signed·positive VJP는0, T1답834token만 PG신호다. 아래 최소시험 진행중 문장은 과거 관측이며 이 종료근거가 우선한다.

**현재 실행:** `ten-functional-v10`, session29954/worker3246026(start65119294). 같은source/protocol의 `verified-functional-repair-exploration-v1`, 원래고정10/순서/freshF0/최대5update·내부14400/외부14700+kill30초다. `ten-functional-v10-launch-main-v1.json` SHA `cb9a815d…`는 실제 제출·입력검증 단계의 근거이며 새초기값/생성·전체10완료 증거가 아니다. 실제초기624해시→진행원장→종료/저장/새reload 순서로 확인한다. 이번 NF4 최소시험의 allocated peak10,145,948,160B를 물리8GB 완전상주로 해석하지 않는다.

둘째문항의 독립 CPU1.782초/exit0은8풀이·4online 의미감사를 그대로 재현했다.
실제 부모→T2에서 e4 H→G를 C→H로 고친1건과 e8 H→G를 G→H로 바꾼 회귀1건이
함께 있다. T3 두자손까지의3관계를 독립교정3건으로 세지 않는다. online11주장은 원래
정상인r1/r2/r3만 지목해180단위 모두0이며 실제e4 수정 주장은 없다. fresh도17/18이다.
주검토는42파일 SHA/size/mode와 실제원문span을 재확인했다. 정본
source10-second-tree-main-v1.json SHAcb2e8977…이다. 이는 실제수정과 자기점검·노트의
단절 근거이지 노트학습 성공/재관찰 인과효과/내부Vision책임 인증이 아니다. 관찰자의
같은그룹 형제를history에 넣은 첫오류는 기록하고 생산자와 같은 그룹시작history로 정정했다.
생산코드/원문/보상은 불변이며 같은3238651의역전파10/12·현재update/result미완료다.

재관찰 통합은 **IMPLEMENTED / CPU TESTED**다. 영향범위115검사/skip0, 독립Astra12검사,
실제cohort2검사와17파일 Ruff·강화범위mypy를 통과했다. 최초332검사의 실패1/오류12/skip2는
보존했으며 전체332 PASS로 바꾸지 않는다. source392개 SHA `4ee164da…`, protocol
`89181dfc…`를 동결했다. 입력169.094초와 실제 기존NF4 save47.774초/check43.903초,
두 생성·68로그확률 차이0·원본접근0을 확인했다. check의 OS exit는 도구handle 만료로
미확인이고 실제 result와PID부재를 근거로 한다. 정본 `p21-readiness-main-v10.json`
SHA `f958b801…`이며 새 학습 F/S/E나 오염전수검사·사람 검토 증거는 아니다.

`minimal-functional-v10` 실제 첫 single/paired2문항·최대1update를 시작했다.
`minimal-functional-v10-launch-main-v1.json` SHA `7bafe4cc…`는 session32350,
timeout3238650/worker3238651(start64707895)·초기 입력재검증 상태를 기록한다.
1024풀이/8192NOTE·24576문맥·내부14400/외부14430초를 유지한다. 완료되지 않은 단계를
통과로 쓰지 않는다. 다음은 실제 원문·전체노트·original token credit·채널별 signed/positive
VJP →624실물저장 →새 프로세스 reload →현재F 확인 후 원래 고정10 탐색 →양경로/대조군/
별도평가다. S는 online+final 합산이므로 최종 전체NOTE 기여는 final을 따로 확인한다.
n1의 기존 usage/generation 소계는 fresh를 포함하지 않으므로 receipt별 비용을 추가한다.

후속 실제 근거: 첫 재관찰→T2 입력의 CPU processor 대조(5.886초)는 전체 부모 token과
이미지·동일 인용의 보존을 확인했다. 첫 문항의 별도 native 의미 재생(1.548초)은 오답7/
문법거절1·최종노트0·모든 학습 계수0을 재현했다. 재관찰7/8과 달리 재시도의 Trace/답은
바뀌지 않았고, 점검한 A→C는 원래 정상이라 교정 보상이 없다. `source10-first-tree-main-v1.json`
SHA `04af2c67…`에 입력 감사·의미 감사·실제 PID를 결합했다. 둘째는 T2/T3 각정답2 뒤 최종
NOTE4개를 생성했다. CPU 입력감사4.041초는 전체8풀이·성공·두이미지·피드백의12455token/
pixel/grid 일치를 확인했다. 네NOTE는 모두stop이나 성공ID누락·claims=[]이며, 실제CPU
0.190초 문법/계수 대조는 둘째T1 답834token만 비영·online/final 원래/적용계수0을 확인했다.
자유lesson 하나는 모든 시도가 정답·검증됐다고 잘못 서술했고, 해당 자유문은 의미검증·PG
대상이 아니다. 형식 누락만 고쳐도 빈 claim을 오류·수정 학습으로 바꿀 수 없다. 같은worker의
역전파를 유지하며 새update/실물저장·freshreload는 아직 미완료다. 주정본
`source10-note-closure-main-v1.json` SHA `caec89a8…`, 실제입력 감사 SHA `70d65834…`다.
별도회로10/3위상은 기존binder의 pin/경로만 바꿔 현재evaluatorv3에 CPU6.999초/exit0으로
재결합했다. 원래12입력 byte·8거부경계·392소스/2348보호파일을 확인했다. 기존 gold노출
AI검토 재사용/humanfalse/newviews0·개발표본이며 모델평가·새오염전수검사는 아니다.
정본 `circuit-inputs-functional-v10/closure-v1.json` SHA `d01cfcfc…`다. 이 입력 준비와
기능 학습 진행을 정확한 종합노트·내부모듈 책임·새문항 성능개선의 증거로 쓰지 않는다.



**2026-10-27 L3 결정 보완 — 동결8 고정10 시작 당시 기록:** P21에 별도 `verified-functional-repair-exploration-v1` 진입을 추가했다. 현재 소스의 실제 minimum update·저장·fresh reload로 F를 확인한 뒤 원래 고정10의 소규모 학습 탐색을 허용하고, S는 그 실행에서 독립적으로 측정한다. 기본 strict의 S 선행 조건과 D00/R11의 수정학습 성공 기준은 보존한다. 보상·생성·자료·분모·MIS·KL·원문·기울기 검사를 완화하지 않으며 S=false를 성공으로 바꾸지 않는다. v6 F를 새 계약의 F로 재사용하는 bridge도 추가하지 않는다. 실제 업데이트/재로딩이 된 뒤에도 S를 미리 요구해 missing repair skill의 학습을 막던 순환 진입 조건을 분리하는 변경이다. 실제 현재 소스의 통합397 CPU검사/skip0, 최소시험 update1·가중치 변경·저장·별도 프로세스624 tensor/optimizer/실제 생성 복원은 확인했다. 최소시험의 S=false/E=false와 노트 오류·수정 관계 신호0은 그대로다. 근거·호출부·시험 범위는 [plan의 결정 보완](plan.md#functional-exploration-20261027)에 기록했다. 아래 카탈로그·원래 설계의 기준일은 그대로이며, 전체 노트·양 경로·고정10 완료·효과를 인증하지 않는다.

이 문서는 연구 보고서이자 구현 계약이다. 구현자는 현재 코드 목록만 보고 예전 실험을 다시 시작하지 말고 D03 결정, D06 입력·출력, D08 학습 식, D10 변경 범위, D13 반례를 먼저 읽는다. `plan.md`는 실행 순서의 정본이고 이 문서는 의미·수식·인터페이스의 정본이다. 모순이 발견되면 임의로 약한 조건을 택하지 않고 해당 작업을 멈춰 두 문서를 함께 수정한다.

### 2026-10-27 실제 종료와 신규 상태 배치 인터페이스

동결8의 고정10은 70개 풀이와24개 최종노트를 수집했지만, 마지막 업데이트 중 외부14700초 한도에서 종료됐다(exit124). 실제 완료 update3과 체크포인트는 보존됐고 정상 `result.json`은 없다. 마지막 중간저장의 cursor6과 원문10개 수집은 서로 다른 증거다. 정상 전체10 재로딩이나 노트 학습 성공으로 표시하지 않는다. 모든 original/applied online·final token 계수는0이다. 아홉째의 유효한 대체 경로와 잔존 관측오류는 같은 단위의 native 오류 수정과 구분한다.

생성 중 사용하지 않는 AdamW 상태를 옮기는 아래 인터페이스를 다음 소스에 추가했다.

| 항목 | 계약 |
|---|---|
| 함수 | `checkpoint.park_adam_moments(bundle: PolicyBundle, optimizer: AdamW, scheduler: LambdaLR, protocol: dict, step: int) -> Iterator[None]` context manager |
| 진입 검사 | 기존 `_validate_optimizer` 재사용. flat step/exp_avg/exp_avg_sq·CPU scalar step·파라미터와 동거하는 moments·지원 device/layout 확인. state↔state/파라미터 storage alias는 전송 전 거부 |
| 본문 | initialized CUDA exp_avg/exp_avg_sq만 CPU에 보관. 원래 GPU tensor 참조를 유지하지 않고 state/key/original device만 기록. CPU step·파라미터·dtype·상태값·RNG·학습식은 보존 |
| 종료 | `finally`에서 원래 device 복원 후 기존 검증. 복원 실패는 전파하여 다음 update/save를 차단. 복잡한 복구나 상태 삭제는 하지 않음 |
| 호출 경계 | actor별 `local_train.train`의 `build_tree`와 `append_rewarded_notebooks`만. 준비·분기·학습·sync·save 전에 복원 |
| 원장 | `launch.optimizer_state_collection_policy = adamw-moments-cpu-during-collection-v1`; 새 source reload에서 정확 비교. 동결8의 과거 실행은 원래 source8로 검증 |
| 실제 근거 | CPU79, 작은 AdamW CUDA4, 독립Astra CPU14/추가모사5 통과. 강화 범위mypy3/Ruff5 통과. 실제 Qwen step3/cursor6의624 optimizer 상태 왕복50.469초·값/장치/가중치/스케줄러/RNG/NF4 보존을 확인. 새 source의 전체 학습/재로딩·속도는 미검증 |
| 수치 경계 | 실제 GPU allocated4,769,837,056→3,613,880,320→4,769,837,056bytes. 두 moment1,155,956,736bytes(약1.077GiB) 감소 실측이다. 측정에 생성/업데이트는 없으며 reserved·속도·긴 생성의8GB 적합성을 인증하지 않음 |

이 최적화는 노트의 의미 학습을 만들지 않는다. 문법실패 부모의 bounded 재시도 확장, 재시도 전 재관찰의 통제 진단, 실제 양성·signed 노트 VJP와 새 문항 검증은 여전히 필요하다. 자세한 실행 순서·반례는 [plan의 후속 상태](plan.md#runtime-close-and-parking-20261027)를 따른다. `optimizer-parking-candidate-v1/hardware-closure-main-v2.json` SHA096b4fbc…와 실제 결과b19c8d88…가 하드웨어 근거이며, 통과를 새 source의 F/S/E로 재사용하지 않는다. v1 진단은 CPU 배치 중 CUDA 장치까지 같아야 한다는 assert로 실패했다. 생산 helper는 바꾸지 않고 내부 CPU 장치·동일값 및 복원 후 원래 전체 signature를 분리 검증했다. 아래 기존 함수 카탈로그는 최초 설계 시점의 snapshot이며 새 helper의 정본 계약은 이 추가 표다.

### 문법 실패 재시도의 구현 계약 — 2026-10-27

| 함수·연결 | 입력과 출력·검증 |
|---|---|
| `rollout.malformed_retry_receipt(sample, response_contract)` | None-grade·보상0·원래 Completion을 검증하고 실제 parser의 문법 오류 code/raw SHA 반환. 유효한 원문인데 grade만 없는 경우는 거부 |
| `rollout.select_repair_parents(samples, k, key, *, native_audits, invalid_responses, response_contracts)` | 기존 native 후보의 순서를 보존. 그 후보가 없을 때만 코드/원문 SHA가 일치한 malformed 부모를 기존 hash 순서로 최대1개 선택. 선택 원장도 반환 |
| `rollout.repair_parent_feedback(parent, actor, prompts, inventory, *, native_audits, invalid_responses, response_contracts)` | 판정된 부모는 기존 피드백, 문법 실패는 오류 코드·NOT JUDGED·원래 length 종료 사실만 전달. gold/교사 풀이 없음 |
| `prompts.repair_retry_text` | 명시적 None-status와 문법 코드/종료 이유 조합만 허용. 정상 status와 malformed 인자 혼합 거부 |
| 생성·재생·평가 | `_build_repair_tree`, `repair_notebook._inspect_tree`, `local_ladder_recovery`가 공통 선택/피드백을 사용. 평가 raw/online 감사는 순번 대신 실제 sample ID로 결합 |
| version | collector v2·feedback v2·evaluation v2·prompt/protocol hash. 과거 source8 승인 자동 재사용 없음 |
| 불변 조건 | result=None/reward0, 모든 원문·토큰·이미지, n2/1024/최대3/seed 식, 기존 미확정 관측/분모/보상. 문법 복구≠native CORRECTED≠S |
| 현재 증거 | worker 연결125/최종61검사, 독립4경계, 주 통합95검사(91CPU·CUDA4skip). 실제 과거 두 UNCERTAIN_REFS의 새 선택·변조 거부 CPU재확인. 새 Qwen T3/학습/reload는 미실행 |

과거 전체10은 여전히 시간한도 부분 종료다. 위 코드가 있다는 이유로 당시 미확정이나 노트 점수를 수정하지 않는다. 다음 진단은 같은 F0/이전 풀이/이미지/seed의 ORIGINAL 재현 뒤 미검증 재관찰을 제공하는 무업데이트 비교이며, 아직 최종노트·학습 효과를 대신하지 않는다.

### 2026-10-28 정리 — 실제 재관찰 비교에서 확인한 것

실험은 원래 source8의 같은 F0·고정 단일/복수 이미지 부모·seed에서 진행했다. 두 ORIGINAL n2의 토큰·원문·종료방식·선택token 로그확률을 실제로 재현한 뒤 같은 VLM의 공개schema/이미지 재관찰1개를 각 n2가 공유했다. 전체687.320초·exit0, 모델 가중치 변경0·정답/교사 풀이 제공0이다.

| 관측 | 범위와 한계 |
|---|---|
| 단일 이미지 | 두 조건 모두 판정 가능한 오답1개·UNCERTAIN_REFS1개. 같은 native 단위의 교정0 |
| 복수 이미지 | 두 조건 모두 정답2개. WITH_FRESH의 한 응답에서 이전 dir[e4] 위반이 COMPLIANT로 바뀐 기회1 관측 |
| 남은 문제 | 해당 응답은 관측17/18이고 dir[e8] 위반이 남음. authored online 단위180개 모두0이며 최종노트 미생성 |
| 결론 경계 | 정답 증가·정확한 전체노트·수정학습 S·자기개선 E를 입증하지 않음. 한 관측의 교정과 새 오류를 함께 감사해야 함 |
| 정본 | private `retry-fresh-observation-probe-v1/gpu-terminal-main-v1.json` SHAb46ba098…; 독립 CPU89.635초로8retry·2fresh 재생 통과. 교정1·회귀1/online180단위0 확인; closure SHAb65536b5… |

이 결과는 모델이 이미지 정보를 다시 읽고 일부 관측을 바꿀 수 있다는 제한된 관찰이며 내부 Vision/Merger/LLM의 책임을 분리한 증거는 아니다. 다음 통합은 미검증 재관찰을 원래 피드백·전체이력 NOTE·원래 생성token 보상과 연결하는 별도 프로토콜로 검토한다. 호스트가 모델의 노트를 대신 채우거나 교정된 사례만 골라 훈련 성공으로 만들지 않는다. source9의 입력 검증265.751초/차단0과 이 source8 진단의 GPU 결과는 서로 다른 증거다.

## 목차

- [D00 목적과 완료의 의미](#D00)
- [D01 현재 구현과 실제 증거](#D01)
- [D02 연구 근거와 채택 범위](#D02)
- [D03 요구사항·결정 원장](#D03)
- [D04 실행 상태와 전체 흐름](#D04)
- [D05 데이터·누설·출처 경계](#D05)
- [D06 원문 문법·자료형·입출력](#D06)
- [D07 오답노트의 의미 검증](#D07)
- [D08 보상·토큰·RL 목적함수](#D08)
- [D09 학습 모듈 선택](#D09)
- [D10 모듈·함수 변경 지도](#D10)
- [D11 실행·메모리·저장·이식성](#D11)
- [D12 실험과 평가](#D12)
- [D13 테스트·반례·수용 조건](#D13)
- [D14 이행·호환·중단 정책](#D14)
- [D15 위험과 해석 한계](#D15)
- [D16 구현자 인수인계 계약](#D16)
- [D17 논문·공식 코드 출처](#D17)
- [D18 현재 함수·클래스 카탈로그](#D18)
- [D19 핵심 알고리즘 의사코드](#D19)
- [D20 최종 승인 체크리스트](#D20)

<a id="D00"></a>
## D00. 무엇을 만들 것인가

만들 것은 **모델이 실제 실패를 점검하고 수정한 행동과, 그 수정을 설명하는 검증 가능한 노트에 보상을 주는 파인튜닝 파이프라인**이다. 사람이 작성한 정답 풀이를 모방시키는 SFT는 넣지 않는다. 모델이 만든 문장도 정답 문장으로 고정하여 cross-entropy로 외우게 하지 않는다.

사용자가 말한 “자기지도 오답노트”의 의도를 다음처럼 구현한다.

1. 동일한 실제 Qwen 모델이 이미지·문제를 보고 풀이와 답을 생성한다.
2. 별도 검증기가 답의 정오와 검사 가능한 관측·연산의 모순을 판정한다.
3. 실패한 실제 응답을 그대로 문맥에 남긴 채, 모델이 오류 점검과 수정 풀이를 순서대로 생성한다.
4. 실제 정답이 나온 경우에도 검사 가능한 풀이 단계가 틀렸는지 별도로 확인한다.
5. 모델은 모든 실제 시도와 성공한 시도를 함께 보고 최종 오답노트를 작성한다.
6. 검증기가 노트의 주장·인용·수정 관계를 확인한다. 실제 생성 토큰에만 RL 보상을 연결한다.
7. 검증된 관측 오류 근거로 Vision 학습 포함 여부를 선택하고, 실제 업데이트·저장·별도 프로세스 복원을 검증한다.
8. 새로운 문항의 첫 시도 성능을 따로 측정한다. 기능 검증 통과를 정확도 향상으로 바꾸어 말하지 않는다.

엄밀한 방법명은 **독립 검증 보상 기반 자기수정 강화학습**이다. 스스로 문장을 생성한다고 해서 통상적인 자기지도학습이 되는 것은 아니다. 비공개 정답·프로그램 검사에서 보상을 받으므로 감독 신호가 존재한다. “추가 SFT와 교사 모델 풀이 모방을 사용하지 않는 RL”이라고 쓰는 것이 정확하다.

### 완료 수준

| 수준 | 필수 관측 | 이것만으로 말할 수 없는 것 |
|---|---|---|
| F: 기능 | 실제 Qwen 생성 → 실제 보상·원래 토큰 → 유한 기울기 → 파라미터 변경 → 저장 → 새 프로세스 복원 | 노트가 정확하거나 성능이 올랐다는 주장 |
| S: 수정 의미 | 실제 native CORRECTED 관계, 원문 증거, 독립 검증, 검증된 양성 부분 VJP와 전체 부호 포함 합산 VJP 모두 비영, 실제 저장 업데이트 | 내부 Vision 단독 책임, 모든 자연어 추론의 정확성 |
| E: 효과 | 사전 고정한 별도 문항에서 첫 시도·회복·노트 정확성·비용 비교 | 작은 표본을 MMMU 전체 개선으로 일반화 |

이번 설계의 구현 완료 목표는 F와 S를 구분하여 검증할 수 있는 구조다. 작은 기능시험에서 정확도가 반드시 상승해야 하는 것은 아니다. E가 개선을 보여 주지 못하면 **기능 구현은 완료될 수 있지만 대규모 학습 효과는 미입증**으로 남는다. 사용자가 별도 승인하기 전 대규모 학습을 시작하지 않는다.

### 먼저 바로잡는 네 가지 표현

- “맞을 때까지”는 무한 반복이 아니라 사전 고정된 최대 3단계 풀이 안에서의 재시도다. 못 맞히면 UNRESOLVED로 끝낸다.
- “답지를 보고”는 검증기가 정답을 비교한다는 뜻이다. 모델 입력에는 gold 문자열·기준 풀이를 넣지 않는다.
- “찍어서 맞혔는가”는 심리 상태를 판정하지 않는다. **정답인데 검증된 과정 오류가 남았는가**를 관측한다.
- “풀파인튜닝”은 이 로컬 설계에서 실행하지 않는다. NF4 기저는 동결하고 Vision/LLM 어댑터와 네 Merger를 학습한다.

<a id="D01"></a>
## D01. 현재 파이프라인에서 이미 된 것과 부족한 것

### D01.1 확인한 정본

| 근거 | 범위 |
|---|---|
| `docs/PROJECT_STATUS.md` 2026-10-26 첫 절 | 최신 V58 실제 업데이트·복원과 의미상 한계 |
| `fine_tuning/notebook_rl/` 실제 소스 | 호출 관계·보상·모델·입력·검증·저장 구현 |
| `fine_tuning/modules.py` | 실제 모델 모듈의 탐색·구획 |
| `fine_tuning/configs/` 및 `fine_tuning/notebook_rl/local_profile.py` | 버전별 계약·로컬 한도 |
| 비공개 `whole-assessment-unit-normalized-v58/functional-training-reload-closure-main-v1.json` | 기존 실측을 연결한 종료 원장 |
| 기존 `research_error_notebook_self_correction_methods_20261026.md` | 직전 연구 정리; 현재 소스와 다시 대조 |

비공개 정본 root는 `/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/`이다. V58 closure SHA256은 `31e40420e4c8df618e3b5eb620ba683ad1d4b3804dd6c785dbe7a27328320e8d`다. 본 보고서의 소스 검토는 GPU 기울기를 다시 계산한 시험이 아니다.

### D01.2 실제 V58 결과

| 항목 | 기존 관측 |
|---|---|
| 모델 | 실제 pretrained Qwen3-VL-4B, NF4 |
| 최소시험 | 고정 2문항, optimizer 1회 |
| 원문 | 풀이 14개, 최종노트 4개, 미검증 초안 12개 |
| 판정 학습 토큰 | 40개: 양성 27개(I 21/U 6), 음성 13개 |
| 판정 전용 합산 VJP norm | L 0.0007283784 / M 0.0001684869 / V 0.0000482031 |
| 실제 변경 텐서 | L252 / M24 / V48 |
| 저장·새 프로세스 복원 | 624 trainable tensor, optimizer/step/cursor, 단일·복수 이미지 forward·짧은 생성 일치 |
| 완전한 종합노트 | 0/4 |
| native 오류→수정 관계 | 각 노트 173개 판단의 관계가 모두 null; 해당 학습 미검증 |
| 최신 OFF-only 경로 | 미검증 |

따라서 “학습 파이프라인이 전혀 작동하지 않는다”는 결론은 맞지 않는다. 현재 성공은 **일부 오류·미확정 판정을 학습시키는 기능**이다. 사용자가 원한 **어떤 잘못을 어떻게 고쳤는지 배우는 기능**은 같은 것으로 볼 수 없다.

### D01.3 실제 호출 순서가 만드는 차이

```text
local_run.capture / local_train.train
  build_tree                    # T1 → T2 → T3 풀이부터 완료
  append_rewarded_notebooks     # 그 후 전체 이력 노트 생성
  bind_objective
  partition_trees
  optimizer_step
  synchronize / checkpoint / reload
```

현재 종합노트는 재시도 **후**에 만들어진다. 앞서 일어난 수정을 노트가 유발했다고 주장할 수 없다. 일반 구형 프롬프트에는 Error check가 이미 있었지만, 최신 native 프롬프트는 별도 Error check 산문을 금지한다. 이 차이를 무시한 프롬프트 한 줄 수정은 trace·관측·답 추출 계약을 깨뜨린다.

현재 재시도는 선택한 부모 응답과 그 조상의 실제 토큰을 보존한다. 모든 형제 분기를 매 재시도에 넣는 것은 아니다. 최종노트의 전체 이력에는 형제·중간·실패·성공 응답을 모두 넣는다. 두 범위를 구별하여 유지한다.

### D01.4 반복 실패에서 얻은 설계 근거

- 모델이 전체 원문을 받았다는 사실과 정확한 오류 관계를 작성한다는 사실은 다르다.
- 성공한 최종답에도 잘못된 관측·거꾸로 적용한 관계가 남았다.
- 모든 관측·연산을 한 번에 분류하게 해도 관계를 null로 두면 수정 학습이 생기지 않는다.
- 후보 네 개의 보상이 같으면 평균 중심화 후 상대 신호는 0이다. 계수만 크게 해도 없던 차이가 생기지 않는다.
- 짧은 참조형 출력 자체는 과거 `referenced_notebook.py`에서 이미 시도했다. 이번 차별점은 **온라인 오류 점검 뒤 실제 수정 풀이를 내고 그 대응을 검증하는 것**이다.
- 기존 원문·검증기·학습·복원 장치는 재사용한다. 새 훈련 프레임워크로 갈아타는 것은 이 문제의 최소 해결이 아니다.

<a id="D02"></a>
## D02. 관련 연구: 무엇을 인용하고 무엇을 제외하는가

2026-10-26 공개 논문·저자 저장소를 확인했다. 아래 방법을 조합한 우리 설계 전체가 이미 검증되었다는 뜻은 아니다.

| 연구 | 확인한 핵심 | 채택할 부분 | 금지·차이·실행 한계 |
|---|---|---|---|
| SCoRe | 모델 자신의 다회 응답으로 온라인 RL; 두 단계 학습 | 실패 이력에서 수정 행동 학습, 시도별 보상 분리 | Gemini 수학/코딩 결과이며 Qwen3-VL 노트 실험이 아님; 저자 실행 코드는 확인하지 못함 |
| DeepSeekMath/GRPO | 동일 문제 후보의 상대 보상, old/reference·clipping | 같은 부모 상태의 후보 그룹, 실제 생성 토큰 likelihood | 우리 평균 중심화·단위별 마스크는 별도 변형; 원 논문의 std 정규화와 같다고 쓰지 않음 |
| Reflexion | 언어적 반성·메모리를 다음 시도에 활용 | 실패·수정 이력의 명시적 전달 | 가중치 학습 증거가 아니므로 단독으로 파인튜닝 구현 근거가 되지 않음 |
| Self-Refine | 자기 피드백 후 반복 수정 | 생성 순서와 피드백 단계의 구분 | 추론 절차이며 새 가중치 업데이트가 없음 |
| VRRL | 시각 피드백 기반 수정 RL, Qwen3-VL-4B 코드 존재 | 실패 prefix와 이후 수정 행동의 학습 분리 | 공개 전체 경로가 Multi-SFT에서 시작; navigation SFT에는 교사 궤적, 그대로 사용 불가 |
| VL-Rethinker | 자기 검토를 유도한 RL | 수정 행동을 명시적으로 생성할 기회 제공 | 성공한 재검토 궤적에 SFT 보조손실도 사용하므로 전체 recipe는 제외 |
| Vision-SR1 | 시각 설명과 답변을 나누고 지각 보상 구성 | 최종답과 시각 정보 품질을 분리 | cold-start SFT 포함; 현재 README의 Qwen3 형식 보상·LoRA 한계 때문에 drop-in 경로 아님 |
| S²R / RISE / STaR / R3V | 자기 생성 자료로 수정·추론 학습 | 실패 분석의 비교 대상으로 인용 | 자기 생성이어도 SFT/NLL 경로가 있어 이번 학습에 직접 채택하지 않음 |
| Qwen PRM lessons | 맞는 답과 올바른 단계는 다름 | 답 정오와 과정 검증 분리 | 새 learned PRM을 지도학습시키지 않음 |
| CoT faithfulness 연구 | 출력된 설명이 내부 원인을 모두 드러내지 않음 | 노트의 자기진단을 내부 모듈 책임으로 해석하지 않음 | 자연어 설명만으로 causal attribution 불가 |
| QLoRA | 동결 4비트 가중치와 학습 가능한 추가 파라미터 | 기존 NF4·LoRA 저장 및 계산 방식 | 원 논문 SFT 실험은 우리 RL 성공이나 8GB 적합성 증거가 아님 |

출처: [SCoRe](https://arxiv.org/html/2409.12917v2), [DeepSeekMath](https://arxiv.org/html/2402.03300v3), [Reflexion](https://arxiv.org/abs/2303.11366), [Self-Refine](https://arxiv.org/abs/2303.17651), [VRRL 논문](https://arxiv.org/html/2607.02490v1)·[공식 코드](https://github.com/fc2869/VRRL), [VL-Rethinker](https://arxiv.org/html/2504.08837v3), [Vision-SR1](https://arxiv.org/html/2508.19652v1)·[구현 한계](https://github.com/zli12321/Vision-SR1), [S²R](https://arxiv.org/html/2502.12853v1), [RISE](https://arxiv.org/html/2407.18219v2), [STaR](https://arxiv.org/abs/2203.14465), [R3V](https://arxiv.org/html/2411.00855v1), [Qwen PRM](https://arxiv.org/html/2501.07301v1), [faithfulness](https://arxiv.org/html/2505.05410v1), [QLoRA](https://arxiv.org/html/2305.14314v1).

### D02.1 왜 보상 항을 처음부터 늘리지 않는가

SCoRe의 두 번째 시도 보너스는 두 번째 시도에 배정된다. 이를 전체 시도에 그대로 전파하면 처음 일부러 틀리고 고치는 경로를 더 높게 보상할 수 있다. 예를 들어 `r1+r2+α(r2-r1)`을 모든 행동에 쓰면 α>1에서 오답→정답이 정답→정답보다 커진다. 이는 **잘못 옮긴 구현의 반례**이고 원 논문의 실패 주장이 아니다. [SCoRe §5.2·Appendix A.2](https://arxiv.org/html/2409.12917v2#A1.SS2)

우리 v1은 추가 성공 차이 보너스를 쓰지 않는다. 검증된 수정 행동과 최종 노트의 원래 토큰에만 분리된 보상을 준다. “새 오류를 찾았다”는 횟수 보상, 장문 보상, 무제한 재시도 보상도 없다. 보상 shaping의 정책 보존 정리는 특정 potential 형태에 대한 것이므로 임의의 오답노트 점수에 자동 적용하지 않는다. [Ng·Harada·Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

### D02.2 논문으로 주장할 수 있는 범위

우리 기여 후보는 “검증된 오류 점검→실제 수정→전체 이력 노트→성분별 RL”의 연결과 검증 원장이다. 성능 결과가 없다면 **설계 및 기능 검증 보고**로 쓰고 새 방법의 우월성을 주장하지 않는다. 다른 연구의 정확도 개선 수치를 우리 모델·데이터·하드웨어에 옮겨 적지 않는다. 저자 코드 존재, 4B 지원, 추가 SFT 여부, 실제 실행 검증은 각기 별도 사실이다.

<a id="D03"></a>
## D03. 요구사항과 고정 결정

| ID | 요구사항 | 승인 기준 |
|---|---|---|
| R01 | 고정 실제 pretrained checkpoint | 모델·revision·quantization·processor hash 일치 |
| R02 | 추가 SFT·교사 모방 금지 | 목표 응답 labels/NLL 경로 없음; 생성 행동의 RL만 |
| R03 | MMMU/MMMU-Pro와 파생물 학습 금지 | 출처 전파·전체 split 대조·미해결 후보 차단 |
| R04 | actor/비공개 검증 자료 분리 | gold·기준 trace·expected issue가 payload에 없음 |
| R05 | 실제 원문 이력 보존 | 부모·형제·실패·성공의 token/text/hash 결합 |
| R06 | 온라인 오류 점검 뒤 실제 수정 | 같은 completion 안의 순서·실제 child witness 확인 |
| R07 | 최종 오답노트 | 모든 시도 목록, 한정된 주장, 누락·거짓 주장 별도 집계 |
| R08 | 독립 의미 검증 | 정답만으로 수정·노트 정당성을 인증하지 않음 |
| R09 | 독립 학습 경로 | ON/OFF/UNRESOLVED, 자기주장 미사용 |
| R10 | 성분별 RL credit | 답·점검·노트·history 마스크 구분 |
| R11 | CORRECTED 합산 기여 | 양·음 전부 포함한 실제 parameter-identity VJP |
| R12 | 실제 변경·저장·fresh reload | 텐서 실물과 실제 단일/복수 이미지 추론 대조 |
| R13 | 명시적 한도 | seed/출력/context/image/update/실행 예산 선고정 |
| R14 | 분리 평가·대조군 | 첫 시도·회복·노트·비용, 노트 보상 제거 대조 |
| R15 | 폴더·캐시 이식성 | fine_tuning만으로 동작, 원본 없는 offline 복원 |
| R16 | MMDL·역사 보존 | 제출·V8·기존 원문·checkpoint 불변 |
| R17 | 자동 클라우드·확대 금지 | 로컬 검증 범위와 별도 승인 단계 구분 |
| R18 | 증거에 맞는 보고 | fixture/실제 기능/수정 의미/성능 효과 구분 |

### 결정 원장

| 결정 | 채택 | 제외한 대안과 이유 |
|---|---|---|
| 생성 단위 | T2/T3 한 completion: ErrorCheck + RevisedSolution | 별도 critique마다 다른 child prompt를 동일 Group처럼 취급하면 상대 보상 계약이 깨짐 |
| 노트 크기 | 전체 이력을 입력하되 최대 3개 핵심 관계 주장 | 173개 정상 판단 의무출력을 반복하여 수정이 생긴다고 가정하지 않음 |
| 완전성 | 정확한 제출 주장과 전체 오류 coverage 분리 | 작은 노트를 기존 strict 전체노트 성공으로 둔갑시키지 않음 |
| 답 보상 | 기존 binary 정오 유지 | 과정 지지 여부까지 답 보상에 섞으면 다른 실험이 되고 무신호 원인이 늘어남 |
| 수정 보상 | 원문에 실제로 생긴 native CORRECTED | 단순 “고치겠다”, PERSISTED, OUTPUT 완성을 수정으로 세지 않음 |
| 라우팅 | 기존 strict_routing 재사용 | 노트의 “Vision 잘못”만으로 모듈 업데이트 선택 불가 |
| 새 파일 | repair_contract.py, repair_credit.py 두 핵심 모듈 | 새 trainer/플러그인/메시지 버스/모델 서버 재설계 불필요 |
| 실패 처리 | 고정 시험에서 신호 없음 기록, 원인별 다음 계약 | 같은 조건을 성공할 때까지 무한 재실행하지 않음 |

### 새 로컬 프로토콜 초안

아래는 **제안값**이다. 기존 결과를 이 설정으로 재표시하지 않는다. CPU 용량 검사에서 문제가 나면 새 실행 전 별도 버전으로 바꾸고 이유를 적는다.

| 항목 | 값·정책 |
|---|---|
| protocol_id | `local-nf4-verified-repair-notebook-v1` — 신규, 아직 구현 아님 |
| model/revision | `Qwen/Qwen3-VL-4B-Instruct` / `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 저장·연산 | NF4 double quantization / BF16 compute / 동결 base |
| context | 24,576, 자동 절단 금지 |
| T1/T2/T3 | n=4/2/2, 최대 3단계 |
| 풀이 cap | 각 1,024; T2/T3의 ErrorCheck까지 **합계**에 포함 |
| 최종 DIAG | n=4, cap8,192; 최대 3관계 주장, 전체 시도 목록 |
| sampling | 현재 로컬 학습 temperature1 / top_p1 / top_k0 유지 |
| seed | base3407 + 기존 문항/정책/단계/분기 해시; 생성 실제 seed 기록 |
| 이미지 | 기존 image config·순서·processor 결과 그대로; 값·hash manifest에 복사 |
| microbatch | 1, 동일 live model 생성/학습 공유 |
| 업데이트 | 최소2문항 최대1회; 고정10 최대5회 |
| optimizer | 기존 AdamW, L1e-5/M1e-6/V1e-6, β(.9,.95), eps1e-8, weight_decay0 |
| KL | 기존 .01; 원래 행동 mask와 frozen NF4 F0 reference |
| 새 단위 보상 가중치 | online repair .0625 / final note .0625 |
| 기존 fact/assessment/strict PG | 새 프로토콜에서는 0; 감사 점수는 계속 기록 |
| progress bonus | 0; 성공 차이 보너스 미사용, 다른 보상 악용 가능성은 별도 검사 |
| 통신 | offline local, RunPod/API/외부 게시 없음 |

1,024 안에 오류 점검과 풀이가 충분히 들어가는지는 아직 검증하지 않았다. 잘림이 많다면 동일 표본·양 arm의 한도를 함께 바꾼 **새 프로토콜**을 사전 등록한다. 오답 문항만 예산을 늘려 기존 결과를 교체하지 않는다. 실제 값은 defaults를 추정하지 않고 생성된 canonical manifest와 대조한다.

<a id="D04"></a>
## D04. 상태 머신과 실행 순서

```text
DATA_BOUND → INPUT_CHECKED → MODEL_READY
  → T1_GENERATED → ANSWER_AND_PROCESS_AUDITED
  → [RETRY_ELIGIBLE and attempts_remaining]
       T2/T3: ErrorCheck → RevisedSolution → FRESH_AUDIT
  → TERMINAL_HISTORY
       answer-correct candidate exists → ALL_HISTORY_FINAL_NOTE → NOTE_AUDIT
       no answer-correct candidate    → UNRESOLVED_RECORD
  → INDEPENDENT_ROUTE
       UNRESOLVED → SAVE_NO_ADMISSION
       ON/OFF → CREDIT_BOUND → PREUPDATE_REPLAY → UPDATE_OR_NO_SIGNAL
  → SYNCHRONIZED → CHECKPOINT_CLOSED → FRESH_PROCESS_RELOAD
  → FUNCTIONAL_RESULT / REPAIR_RESULT
```

### 상태의 의미 — 실제 코드에는 영어 상태 문자열 사용

| 상태 | 입력 | 출력 | 금지 |
|---|---|---|---|
| DATA_BOUND | 승인 범위와 고정 ID·파일 | data/source/protocol 결합 | 다른 문항으로 몰래 교체 |
| INPUT_CHECKED | actor·이미지·토큰 예약 | 실제 input/grid/pixel/token hash | gold 접근·입력 축약 |
| T1_GENERATED | 동일 초기 요청 | 원문 n4 전체 | 정답 후보만 저장 |
| ANSWER_AND_PROCESS_AUDITED | 원문과 비공개 checker | 답 정오와 과정 상태 별개 | 정답이면 과정 전부 정상으로 처리 |
| RETRY_ELIGIBLE | 실패 또는 검증된 모순 | 기존 결정적 failure selection | UNKNOWN을 실제 오류로 전환 |
| FRESH_AUDIT | 원래 parent+새 completion | 수정·지속·미확정 증거 | 이전 verdict를 신뢰해 생략 |
| ALL_HISTORY_FINAL_NOTE | 전 시도·이미지·공개 결과 | 모델 자체 노트 n4 | host가 관계를 대신 작성 |
| CREDIT_BOUND | 고정 원문·원래 그룹 | 실제 token별 계수 | 새 문자열을 행동으로 학습 |
| UPDATE_OR_NO_SIGNAL | 같은 policy batch | update 또는 구체적 무신호 | 동일 보상에 가짜 분산 추가 |
| CHECKPOINT_CLOSED | 완료 update | 독립 저장 payload/manifest | 미완료 저장을 완료로 표기 |

### 종료 규칙

1. 답이 맞고 독립 검증 가능한 모순도 없으면 그 분기의 재시도를 끝낸다. 검사 범위가 부족하면 “답은 맞음/과정 미확정”으로 남긴다.
2. 답이 맞아도 실제 VIOLATED 관측·연산이 있으면 남은 단계 안에서 재시도 대상으로 삼을 수 있다. 원래 답 정오는 CORRECT로 유지한다.
3. 각 단계는 기존 선택 정책으로 한 실패 부모를 고른다. 형제 후보가 맞혔다는 이유로 다른 부모의 회복으로 세지 않는다.
4. 마지막 시도까지 실패하면 문항을 실패로 닫는다. 유효한 부분 연산 수정은 별도 보상 대상일 수 있지만 성공 오답노트를 만들어내지 않는다.
5. CONTEXT_BLOCKED, BUDGET_STOP, OOM, 비유한 손실, 원문 변조는 원인을 저장하고 종료한다. 한도·모델·검증 기준을 자동 완화하지 않는다.

부모 선택의 결정성도 고정한다. 직전 단계의 기존 RETRYABLE 후보와 독립 VIOLATED가 확인된 CORRECT 후보의 합집합에 현재 `select_failures`의 순서를 적용한다. 먼저 `SHA256(f"{key}:{sample_id}")` digest 오름차순, 그 안에서 추출 답의 첫 등장 후보를 앞에 두고 같은 답의 반복 후보를 뒤에 둔다. T2의 key는 기존 `problem_id:policy_version`, T3의 key는 `problem_id:policy_version:2`다. branch는 각각1이며, 새 후보라고 별도 우선순위를 주거나 정답값·노트 품질로 다시 정렬하지 않는다. 후보는 각각 T1/T2 frontier에서만 가져온다.

### 올바른 예와 반례 — 설명용 fixture, 실제 모델 결과 아님

```text
parent: 이미지에 H→G가 있는데 G→H라고 관측함. 최종답은 우연히 맞음.
child ErrorCheck: parent의 해당 ref를 다시 관측하겠다고 지정.
child RevisedSolution: H→G로 관측하고 실제 경로에서도 H→G로 사용.
private check: 같은 edge identity, descendant, original VIOLATED, later COMPLIANT.
결과: answer grade는 원래도 CORRECT; native observation repair는 새로 확인.
```

child가 `H→G`라고 적기만 하고 계산에서는 다시 `G→H`를 사용했다면 관측 수정과 규칙 적용 실패를 동시에 기록한다. 최종답 하나로 둘을 합쳐서 성공시키지 않는다.

<a id="D05"></a>
## D05. 학습 자료와 누설 방지

### D05.1 현재 실제 자료

| 자료 | 확인된 구성 | 사용 가능 주장 |
|---|---|---|
| local train pool | 16문항/14구조, graph·graph-pair | 기존 제한 로컬 시험의 입력 후보 |
| 고정 train10 | BFS2, DFA2, preorder tree1, paired shortest-path5 | 엔진·수정 기능 시험, 15이미지 |
| 기존 dev/sealed/style | 각8문항/7구조 | 선택된 구조 ID 분리, 출처·계열은 공유 |
| 별도 회로 eval10 | 이상적 DC ladder, 3개 topology 계열 | 새로운 bounded 회로 개발 평가 |

현재 자료를 “4개 전문 실패군 전체를 대표하는 학습 데이터”라고 쓰지 않는다. 같은 generator를 다른 seed로 만든 것은 새로운 출처가 아니다. 기존 회로10은 개발 중 확인한 자료이며 앞으로도 영구 sealed라고 주장할 수 없다.

기존 오염 원장은 `EXCLUSION_CHECKED_WITH_LIMITS`, 40문항을 대조했고 미해결 blocker가 없다고 기록한다. 이는 새 전수 스캔을 이번에 한 것이 아니며, 원장 자체의 `approval_granted=false`, `training_allowed=false`, `human_verified=false`도 보존한다. **기존 제한 시험 실행 권한과 데이터셋 전체 학습 승인은 별개**다. 새 프로토콜은 기존 제한 허용 범위·입력 gate를 재결합해야 한다.

### D05.2 Actor와 private verifier

| 필드 | Actor 입력 가능 | 검증기 전용 |
|---|---|---|
| 질문·보기·실제 이미지 | 예 | 동일 identity 확인 |
| 모델이 쓴 이전 풀이·답·ErrorCheck | 예 | 그대로 독립 재채점 |
| 정오 피드백 | 고정된 제한 표현만 | 상세 근거 계산 |
| 공개 task grammar·일반 연산 규칙 | 예 | 버전 결합 |
| 정답 gold·gold explanation | 아니오 | 예 |
| 완전한 scene graph·정답 경로·기준 trace | 아니오 | 예 |
| private expected issue·corrected value | 아니오 | 예 |
| 원문 위치의 공개 ref ID | 예 | 원문과 identity 검증 |
| prior answer와 later answer 비교 | 모델 자신의 답은 가능 | gold 동일성은 비공개 |

정오 피드백에는 실제 정답 문자열이 없어도 감독 신호가 있다. “감독 없음”이라고 보고하지 않는다. 틀린 slot ID, 정답값, 완성된 풀이를 피드백에 추가하려면 별도 실험 조건으로 등록해야 한다. v1은 답 정오/과정 모순 유무만 알려주고 상세 오류 위치는 모델이 고른다.

### D05.3 출처 전파 및 겹침 검사

1. 원자료 ID, repository revision, split, 라이선스·재배포 조건, 파일 SHA, 부모 ID를 lineage에 고정한다.
2. MMMU/MMMU-Pro의 어떤 split도 train source로 받지 않는다. 문항 전사·번역·크롭·재렌더링·교사 설명도 BENCHMARK_DERIVED다.
3. benchmark-derived parent가 하나라도 있으면 `training_allowed=false`가 자손에 전파된다.
4. 텍스트 exact·정규화·근사, 이미지 byte·pixel·perceptual 후보, 부모 출처를 대조한다. 해시 후보는 동일 문항 확정이 아니다.
5. 사용자가 정한 정책대로 **미확인 중복 후보가 있으면 중단**한다. 검출기 완화·후보 자동 무시를 하지 않는다.
6. 공개 시험/교과서 원천 중복까지 완벽히 배제했다고 주장하지 않는다. 검사 버전·비교 corpus·경고·검토자를 남긴다.
7. AI 검토와 사람 검토를 구분한다. 실제 사람이 확인하지 않았으면 human_verified는 false다.
8. 학습과 평가의 source/template/structure/image/near-duplicate 군을 따로 비교한다. 일부 분리가 다른 분리를 대신하지 않는다.

### D05.4 네 실패군을 위한 후속 데이터 순서

| 실패군 | 우선 자료 | private verifier에 필요한 것 | 현재 상태 |
|---|---|---|---|
| A 전문 기호 | 독립 생성 악보·기호, 추후 원본 MUSCIMA++/기하 자료 검토 | 위치·기호·연결 native truth, 실제 음정/관계 규칙 | 현 graph 검증기로 대체 불가; 신규 검증기 필요 |
| B 도식·공간 | 현 그래프 → CLEVR → AI2D | scene graph·실행 가능한 프로그램·관계 annotation | 그래프만 현재 bounded 구현; 나머지 후보 |
| C 과학 판독 | AI2D bounded 도식, 독립 시뮬레이션 | 판독 가능한 속성·도식 관계와 검증 범위 | 질환·전문 판단 전체를 프로그램으로 인증할 수 없음 |
| D 수량 추론 | 현 경로·회로 → TabMWP | native 수치·단위·연산·필수 입력 연결 | TabMWP adapter 존재, 승인/필수 slot 범위 미완료 |

후속 자료를 각1000개씩 곧바로 섞지 않는다. 한 자료군의 actor/private 변환·중복 검사·renderer 검토·연산 checker·노트 반례가 통과한 뒤 사전 고정한 수량을 추가한다. 검증기가 없는 자료를 억지로 “시각 정상” 판정하여 OFF로 학습시키지 않는다.

CLEVR의 scene과 functional program은 private checker 후보로 유용하다. 정답 프로그램을 모델의 target trace로 주면 별도 지도 신호가 되므로 이번 정책에서 제외한다. AI2D의 도식 주석은 제한된 관계 검사에 쓸 수 있지만 자유 과학 풀이 전부의 정답은 아니다. TabMWP의 풀이 문자열도 actor/target에서 제외한다. [CLEVR 공식 배포](https://cs.stanford.edu/people/jcjohns/clevr/), [생성 코드](https://github.com/facebookresearch/clevr-dataset-gen), [AI2D 논문](https://arxiv.org/abs/1603.07396), [TabMWP 공식 코드](https://github.com/lupantech/PromptPG)

<a id="D06"></a>
## D06. 새 생성 문법과 자료형

**이 절의 신규 이름은 제안 API다. 현재 존재하는 함수 목록은 D18에 별도로 있다.** 구현은 표준 dataclass·기존 strict JSON 도구를 사용하고 새 schema framework를 도입하지 않는다.

### D06.1 원문 재시도 문법

```text
ErrorCheck: {"version":"repair-check-v1","claims":[...],"uncertain_refs":[...]}
RevisedSolution:
Observed facts:
  ...기존 native 관측 문법...
Solution:
Trace: ...기존 task별 trace JSON...
Final answer: ...모델 자신의 답...
```

규칙:

- 정확히 한 개의 선두 ErrorCheck JSON과 한 개의 RevisedSolution 경계만 허용한다.
- JSON 문자열 안의 문자나 인용된 `Final answer:`를 구조 경계로 오인하지 않는다.
- marker는 시작 위치와 줄 경계를 이용하여 파싱한다. `str.split()`의 무제한 전역 치환을 쓰지 않는다.
- UTF-8 파일 byte SHA와 Python 문자열 char offset을 구분한다. span은 원문 문자열의 반열린 구간 `[start,end)`다.
- 원문을 Markdown 제거·재직렬화·재토큰화하여 새 행동으로 만들지 않는다.
- 값 파싱은 JSON bool을 int로 인정하지 않고 NaN/Infinity/중복 key·알 수 없는 필드를 거부한다.
- ErrorCheck의 과거 오류 인용은 새 관측·Trace·최종답이 아니다. checker에는 RevisedSolution 범위만 투영한다.
- projection report의 모든 span·인용 hash는 원문 좌표로 다시 올린다. 최종 inventory·token credit·replay가 같은 매핑을 사용한다.
- 종료가 length라면 완성 unit 여부·답 종료 여부·전체 응답 잘림을 분리한다. 완성되지 않은 claim은 학습하지 않는다.
- 기존 프로토콜에서는 기존 parser를 그대로 사용한다. 새 문법을 legacy 응답에 소급 강제하지 않는다.

### D06.2 ErrorCheck 주장

```json
{
  "version": "repair-check-v1",
  "claims": [
    {
      "origin_ref": "public-reference-id",
      "intent": "RECHECK_OBSERVATION",
      "rule": "DIRECTED_EDGE",
      "proposed_value": "model-authored value"
    }
  ],
  "uncertain_refs": []
}
```

값은 예시 placeholder다. 실제 학습 입력에 정답값을 채운 예시를 넣지 않는다. `intent`는 RECHECK_OBSERVATION/RECHECK_OPERATION/UNKNOWN 중 하나다. 최대3 claim, 같은 origin 중복 금지. `rule`은 해당 task가 공개한 규칙명만 허용한다. `proposed_value`는 observation의 값 또는 task별 typed operation이며 UNKNOWN일 때 null이다. 과거 결과의 정답을 복사하는 필드는 없다.

모델이 “다시 보겠다”라고만 쓰면 아직 수정 성공이 아니다. **그 뒤 같은 completion의 실제 RevisedSolution**에서 일치하는 수정값·연산을 제출해야 native repair 보상을 받을 수 있다. 새로운 관측으로 다른 오류를 만든 경우도 함께 감사한다.

### D06.3 최종 노트 문법

```json
{
  "version": "repair-note-v1",
  "scope": "BOUNDED_CLAIMS_FULL_HISTORY",
  "attempts": [
    {"sample_id":"actual-id","outcome":"CORRECT","answer":"own-answer"}
  ],
  "success_sample_id": "actual-successful-id",
  "claims": [
    {
      "origin_ref":"actual-earlier-ref",
      "later_ref":"actual-later-ref",
      "relation":"CORRECTED",
      "rule":"DIRECTED_EDGE",
      "repaired_value":"model-authored-value"
    }
  ],
  "uncertain_refs": [],
  "lesson_text": "optional model-authored explanation"
}
```

- attempts에 실제 풀이의 모든 sample_id가 원래 순서로 정확히 한 번씩 들어가야 한다. DIAG/미검증 초안은 풀이 시도 수에 넣지 않는다.
- 각 outcome/answer는 해당 원문에서 추출한 자기 답과 공개 정오의 일치 여부를 검사한다. host가 빈 항목을 채우지 않는다.
- success_sample_id는 실제 CORRECT가 존재하는 가장 마지막 풀이 단계에서 원래 sample 순번이 가장 작은 시도다. 정답 값이나 노트 품질로 선택하지 않는다. 그 시도에 과정 모순이 남아 있어도 CORRECT 판정은 보존하고 별도 속성으로 보고한다. 성공이 전혀 없으면 성공 노트를 만들지 않지만, 독립적으로 확인한 온라인 부분수정의 RL은 허용한다.
- claims는 최대3개다. CORRECTED/PERSISTED/OTHER_BRANCH_COMPARISON/UNKNOWN/FINALIZED를 구분한다.
- CORRECTED와 PERSISTED의 later_ref는 실제 descendant를 가리켜야 한다. 형제 비교는 별도 유형이다.
- lesson_text는 사람이 읽는 설명이며 **자유 의미 검증·PG 보상 대상에서 제외**한다. 정답 설명처럼 무조건 학습하지 않는다.
- host의 coverage 통계는 모든 known issue와 제출 주장 차이를 계산한다. 이 비공개 기대 목록을 모델에게 보내지 않는다.
- 전체 이력 입력을 유지하면서 일부 주장만 작성하는 계약이다. 이를 “모든 오류를 정확히 정리했다”고 부르지 않는다.

#### 관계별 필드와 UNKNOWN 지원 범위 — 엄격한 JSON 계약

각 claim의 공통 필드는 위 예시 그대로다. 아래 표에 명시한 UNKNOWN 전용 필드만 추가 허용하며, 그 외 임의 필드는 거부한다. `later_ref`는 최종 claim에서 항상 실제 공개 ref 문자열이고 현재 노트·미래 응답을 가리킬 수 없다.

| 최종 relation | `repaired_value` | 추가 필드 | 독립 검사와 보상 |
|---|---|---|---|
| CORRECTED | 실제 후속 연산/관측의 모델 작성 값 | 없음 | 이전 VIOLATED→동일 의미 descendant COMPLIANT·제출값 일치, +1 |
| PERSISTED | null | 없음 | 이전 VIOLATED→동일 의미 descendant에서도 VIOLATED, +1; 수정 성공 아님 |
| UNKNOWN | null | `unknown_reason`, `blocked_by` 필수 | 아래 좁은 의존성 계약을 모두 만족할 때만 +1 |
| OTHER_BRANCH_COMPARISON | null | 없음 | 같은 문제·공유 조상인 비-descendant 분기, 원래 VIOLATED와 같은 의미의 다른 분기 COMPLIANT 대조, +1; 인과적 수정 아님 |
| FINALIZED | null | 없음, rule은 FINALIZATION | 답 부재→실제 descendant의 완성 답; online/final 보상0·별도 기록 |

이번 버전의 `unknown_reason`은 공개 상수 `DEPENDENCY_BLOCKED` 하나만 지원한다. `blocked_by`는 **모델이 제출한 public ref 문자열 배열**이며 1개 이상·중복 금지·공개 inventory 순서다. 각 ref는 later_ref와 같은 실제 후속 시도의 선행 연산을 가리킨다. 검증기는 원래 오류와 같은 의미의 후속 연산이 실제 UNVERIFIABLE인지, 제출 dependency가 native checker의 `blocked_by`와 누락·추가 없이 일치하는지, 그 선행 위반의 근거 span이 맞는지 확인한다. 이전 오류와 실제 descendant 관계도 필수다. host는 ref를 실제 rule/locus/span으로 풀어 쓸 수 있지만 **누락한 blocker나 reason을 대신 작성하지 않는다**.

root trace 형식 불가·관측 누락·근거 없는 일반적 “잘 모르겠다”는 이 강한 UNKNOWN 보상 계약을 만족하지 않는다. 그런 제출은 UNVERIFIABLE/점수0이며 실패 원장을 유지한다. 이후 지원 확장은 새 schema/verifier version과 반례를 필요로 한다. UNKNOWN 이외 claim에 unknown_reason/blocked_by가 붙으면 STRUCTURAL_INVALID다. ErrorCheck의 `intent=UNKNOWN`은 점검 보류 의도이며 `proposed_value=null`, 전용 reason/blocker 필드 없이 online0이다. 그것을 최종 typed UNKNOWN +1과 혼동하지 않는다.

양쪽 문법의 `uncertain_refs`는 `list[str]`, 최대3개·고유한 실제 공개 ref·inventory 순서이며 claims의 origin과 중복하지 않는다. 아직 판단하지 않은 항목을 표시하는 필드로 **보상0**이다. 그 자체를 참/거짓 claim precision의 분모에 넣지 않고 declared_uncertain count·미커버리지로 별도 보고한다. 이 필드가 비었다고 모든 항목을 검증했다고 해석하지 않는다. 정확한 제출 노트의 unverified=0 조건은 제출한 의미 claims에 대한 조건이며 전체 이력의 불확실성이0이라는 뜻이 아니다.

### D06.4 신규 내부 자료형 계약

| 타입 | 필드 | 불변식 |
|---|---|---|
| RawSpan | start:int,end:int | 0≤start≤end≤len(raw), byte offset 아님 |
| RetrySections | raw_sha,check_span,solution_span,claim_spans,contract_version | 원문 기준·중복 경계 없음 |
| PublicOrigin | ref_id,sample_id,kind,rule,locus,occurrence_spans | text/lineage에서만 생성, private truth 없음 |
| RepairClaim | origin_ref,intent,rule,proposed_value,raw_spans | 실제 모델 제출, 최대3·중복 금지 |
| ParsedRepairNote | version,scope,attempts,success_sample_id,claims,uncertain_refs,lesson_text,raw_sha,raw_spans | D06.3 union 필드·전체 시도 순서·원문 소유권 |
| RepairWitness | origin_ref,later_ref,semantic_identity,relation,claim_verdict,evidence_hashes | checker 전용, truth·ancestry·동일 연산 결합 |
| RepairAudit | parent_id,child_id,sections,origins_digest,witnesses,unit_scores,source_bindings | actor에 전달 금지·private 검사와 실제 raw 결합 |
| UnitScore | origin_ref,value,reason,token_spans,verifier_version | value∈{-1,0,1}, 결측 raw와 값0 구분 |
| ComponentCredit | component,coefficients,original_group_size,fixed_units,cap,mask_hash | 토큰 수 일치, finite, 변조 시 실패 |
| NoteAudit | submitted,verified,false,unverifiable,uncovered,corrected,persisted,scope | precision과 coverage 별도, 분모0은 N/A |
| RepairGradientEvidence | parameter_ids,signed_norms,positive_verified_norms,retained_positive,source_bindings | 전체 부호 합산과 양성 검증 부분을 각각 측정, 채널 명시 |

이 타입들은 학습 checkpoint의 기존 tensor 형식을 바꾸지 않는다. 새 protocol metadata와 별도 원장에 필요한 구조다. JSON wire는 strict serializer로 변환하고, runtime dataclass 객체를 pickle 인터페이스 계약으로 삼지 않는다.

### D06.5 신규 API 최소 집합

```python
# NEW: repair_contract.py — private gold를 받지 않는 순수 함수
parse_retry_sections(raw: str, *, contract_version: str) -> RetrySections
parse_error_check(raw: str, sections: RetrySections) -> tuple[RepairClaim, ...]
parse_final_note(raw: str, *, contract_version: str) -> ParsedRepairNote
public_origins(samples: list[Sample], *, contract_version: str) -> tuple[PublicOrigin, ...]
project_solution(raw: str, sections: RetrySections) -> str
lift_solution_spans(report: dict, sections: RetrySections, raw: str) -> dict

# NEW: repair_credit.py — 검증기 측 함수; actor payload를 반환하지 않음
audit_online_repair(parent: Sample, child: Sample, actor: ActorItem,
                    verifier: Any, origins: tuple[PublicOrigin, ...],
                    sections: RetrySections) -> RepairAudit
audit_final_note(sample: Sample, tree: ProblemTree, actor: ActorItem,
                 verifier: Any, origins: tuple[PublicOrigin, ...]) -> NoteAudit
repair_group_credit(group: Group, audits: list[RepairAudit | NoteAudit],
                    tokenizer: Any, *, completion_cap: int,
                    component: str) -> list[ComponentCredit]
```

신규 `ParsedRepairNote`/`RepairAudit`는 위 필드 계약을 갖는 dataclass로 구현한다. `Any`는 현재 verifier/tokenizer 인터페이스를 연결하는 설계 표기이며 임의 객체를 무검사로 신뢰한다는 뜻이 아니다. 구현 시 기존 호출부의 실제 타입으로 좁히고, 타입을 넓히거나 예외를 삼켜 연결 오류를 숨기지 않는다.

실패는 명시적 ValueError 계열 계약 오류로 제어층에 전달한다. malformed 모델 출력을 시스템 crash로 취급하지 않도록 raw 저장 후 audited failure로 바꾸되, 프로토콜·source·token 변조는 모델 오답 점수0으로 숨기지 않고 run integrity 실패로 중단한다.

<a id="D07"></a>
## D07. “틀림을 역추적했다”는 것을 어떻게 검증하는가

### D07.1 기존 검증기 재사용

`reasoning_trace.check_trace`는 native graph world와 모델의 typed trace를 대조한다. `circuit_trace`는 정해진 회로·단위·연산을 검사한다. `notebook_synthesis._lesson`에는 동일 operation identity와 ancestry를 확인하는 로직이 이미 있다. 이 의미를 재사용하는 작은 public wrapper를 두되 기존 strict 전체노트 성공 조건을 online repair에 가져오지 않는다.

현재 `_lesson()`의 `verified=False`에는 거짓·미검증·잘못된 구조가 함께 들어 있다. 이를 그대로 -1로 변환하면 안 된다. 새 `claim_verdict`는 `VERIFIED`, `REFUTED`, `UNVERIFIABLE`, `STRUCTURAL_INVALID` 네 상태와 구체적인 reason/evidence를 반환한다. 독립 근거로 반증된 주장만 REFUTED/-1이다. `RECHECK_OBSERVATION`/`RECHECK_OPERATION`은 점검 의도이므로 그 자체를 거짓 비난으로 세지 않는다. 올바른 수정 제안도 실제 자식 연산을 고치지 않았다면 온라인 수정 보상은0이다. 정상 관측을 잘못됐다고 단정한 최종노트는 기존 `_lesson(issue=True)`에 양성 endpoint가 없더라도 별도 반증 경로로 REFUTED 처리한다.

| 주장 | 필요한 근거 | 불충분한 근거 |
|---|---|---|
| 관측 오류 | 원래 관측 span과 native image fact 불일치 | 모델의 “잘못 봤다” |
| 규칙 적용 오류 | 관측/전제 검증 뒤 실제 operation VIOLATED | 최종답 오답만 |
| CORRECTED | 이전 VIOLATED → 같은 의미 연산의 descendant COMPLIANT | 나중 답만 맞음, 다른 node의 같은 index |
| PERSISTED | 같은 오류가 실제 후속 시도에 남음 | 노트의 반복 주장 |
| 최종 UNKNOWN 주장 | D06.3의 DEPENDENCY_BLOCKED·같은 의미의 descendant UNVERIFIABLE·모델 제출 blocker refs와 checker 의존성의 정확한 대응 | 일반 형식/coverage 부족만으로 +1, Vision 정상 또는 단정적 오답 |
| FINALIZED | 이전 답 부재 → 후속 완성 답 존재 | 해당 답의 정확성·시각 수정 |
| OTHER_BRANCH_COMPARISON | 실제 형제 분기 비교 | 시간상 뒤라는 이유만으로 correction |

### D07.2 답과 과정의 독립 행렬

| 답 | 검사 과정 | 보고 |
|---|---|---|
| 맞음 | 지원 범위 내 모순 없음·coverage 충분 | ANSWER_CORRECT_PROCESS_SUPPORTED |
| 맞음 | 확인된 오류 있음 | ANSWER_CORRECT_PROCESS_CONTRADICTED |
| 맞음 | trace/관측 누락·지원 불가 | ANSWER_CORRECT_PROCESS_UNVERIFIED |
| 틀림 | 검사 가능한 과정 오류 있음 | ANSWER_WRONG_PROCESS_CONTRADICTED |
| 틀림 | 원인 부족 | ANSWER_WRONG_PROCESS_UNVERIFIED |

위 문자열은 audit 속성이다. 기존 Status enum을 바꾸지 않는다. `SUPPORTED`도 전체 자연어 풀이·내부 사고의 인증이 아니라 명시된 checker 범위에 한정한다.

### D07.3 노트 정확성·완전성의 분리

- **submitted claim precision** = 검증된 참 주장 / 제출한 의미 주장. 미검증·거짓·잘못된 ref도 제출 분모에서 사라지면 안 된다.
- **known issue coverage** = 올바르게 언급한 고유 known issue / 독립 checker가 찾은 고유 known issue.
- **native correction coverage** = 모델이 올바르게 연결한 실제 native correction / 독립 감사에서 확인된 correction opportunity.
- **false accusation count** = 실제 정상 관측·연산을 잘못됐다고 주장한 수.
- **unverified count** = 참·거짓 어느 쪽도 확인 못한 제출 주장.
- **uncovered count** = 확인된 오류 중 노트가 다루지 않은 수.
- 분모가 0이면 N/A와 실제 count를 기록한다. 빈 lessons에 all()=true가 나와도 노트 성공으로 세지 않는다.

`accurate_submitted_note=true`는 1개 이상 검증된 의미 주장, false=0, unverified=0, 전체 시도 목록/성공 연결 일치를 모두 요구한다. 이 값과 `complete_known_issue_coverage`는 별개다. 최대3 claim 때문에 넓은 이력에서 후자가 false인 것은 허용되는 **명시된 기능 범위**이며, 전체노트 완성으로 바꿔 보고하지 않는다.

### D07.4 검증 가능 설명의 범위

검증 가능한 교훈은 “이 시도의 이 관계를 뒤집어 사용했고, 해당 후속 시도에서 같은 관계를 올바르게 사용했다”처럼 닫힌 증거에 연결한다. “나는 그림 이해 능력이 부족하다”, “앞으로 항상 신중하자” 같은 자유 문장은 저장할 수 있지만 사실·원인 보상으로 쓰지 않는다.

독립 프로그램은 이미지에 그려진 세계와 annotation이 일치하는지까지 자동으로 보장하지 않는다. 생성기 self-check·renderer 검토·실제 입력 해시가 필요하다. gold와 연산 checker가 같은 버그를 공유할 가능성도 별도 solver 또는 손검산 fixture로 검사한다.

<a id="D08"></a>
## D08. 정답·오답노트가 실제 가중치 학습으로 들어가는 방법

### D08.1 입력, 행동, 보상의 구분

문제·이미지·이전 풀이·피드백은 조건 입력이다. 새로 생성한 ErrorCheck/수정 풀이/최종노트가 행동이다. gold는 검증기에만 있고 학습 모델에 target sentence로 넣지 않는다. 최종 정답은 **정오 보상의 근거**이며 외워야 할 teacher completion이 아니다.

역전파는 해당 문맥에서 실제 생성 토큰의 log probability를 미분한다. NF4 저장 기저는 동결되지만 그 계산을 통해 Vision/LLM 어댑터와 Merger에 기울기가 흐를 수 있다. LoRA는 역전파의 대체물이 아니라 학습 파라미터를 줄이는 표현이다.

### D08.2 정책 세 개

| 정책 | 실제 구현 |
|---|---|
| sampler/current | 같은 live unmerged PEFT 모델; 생성 후 같은 policy에서 업데이트 |
| old | 업데이트 전에 같은 모델로 no_grad logprob를 재계산하여 CPU에 보관 |
| reference | adapter 비활성 + 원래 frozen Merger를 선택한 동일 NF4 F0 |

새 4B 모델 세 개를 동시에 적재하지 않는다. reference가 BF16 원본이라고 쓰지 않는다. old/current/source/policy mismatch는 재사용을 거부한다. 여러 optimizer epoch를 같은 rollout에 무단 반복하지 않는다.

### D08.3 토큰 소유권

| 영역 | answer PG | online repair PG | final note PG | KL/MIS |
|---|---:|---:|---:|---:|
| 질문·이미지·이전 시도 | 0 | 0 | 0 | action mask0 |
| T1 새 풀이 | 기존 binary | 0 | 0 | 기존 정책 |
| T2/T3 ErrorCheck | **0** | 검증된 claim literal만 | 0 | 원래 생성 action |
| T2/T3 RevisedSolution | 기존 binary | 검증된 실제 수정 value/operation만 | 0 | 원래 생성 action |
| DIAG 참조·관계·수정값 | 0 | 0 | 독립 검증한 원래 literal만 | 원래 생성 action |
| DIAG 자유 lesson_text | 0 | 0 | 0 | 기존 KL만 가능 |
| host 확장 JSON·정답 trace | 0 | 0 | 0 | 행동 아님 |

“다른 채널”이라는 변수명만으로 분리되지 않는다. scalar answer advantage가 ErrorCheck에 broadcast되지 않도록 실제 token vector에서 0이어야 한다. 같은 token이 여러 claim에 중복 할당되면 double counting을 거부한다. 같은 operation의 claim span과 실제 수정 span에 보상을 나눌 때 전체 합이 한 단위가 되도록 토큰 집합의 **합집합을 한 번 정규화**한다.

이 분리는 해당 토큰의 log-probability에 붙는 **직접 계수**의 분리다. 뒤쪽 수정 풀이의 손실은 앞 문맥을 사용한 계산과 공유 파라미터에 영향을 주며 ErrorCheck에도 KL은 남는다. 따라서 ErrorCheck의 answer PG가0이라는 이유로 정답 손실과 완전히 인과적으로 격리됐다고 주장하지 않는다.

### D08.4 단위 점수와 고정 분모

한 group g의 원래 후보 수를 n, 공개 origin inventory 수를 U, candidate i가 origin u에 대해 낸 점수를 r(i,u)라 한다. 공개 inventory는 모델 원문과 lineage만으로 생성하여 모든 후보에게 같게 고정한다. private한 실제 오류만 골라 분모를 줄이지 않는다.

각 그룹을 생성하기 **전** 그 그룹이 공유하는 공개 부모 이력으로 origin 집합·순서·digest를 고정한다. 정상·오류 미확정 항목도 포함한다. 최종 DIAG에서는 이미 닫힌 전체 풀이 이력을 사용한다. 현재 `whole_reference_credit.check()`의 private VIOLATED 목록 기반 unit 생성은 재사용하지 않는다. 재사용 대상은 안전한 원문 토큰 매핑뿐이다. 기존 private 오류 목록을 그대로 쓰면 정상 항목에 대한 거짓 비난이 학습 원장에서 사라진다.

| 제출 상태 | online repair | final typed note |
|---|---:|---:|
| 독립 검증된 native CORRECTED + 실제 일치 patch | +1 | +1 |
| 참인 PERSISTED/지원되는 UNKNOWN/형제 비교 | 0 | +1 |
| OUTPUT FINALIZED | 0 | 0: 별도 기록 |
| 분명히 거짓인 semantic claim | -1 | -1 |
| 미제출·검증 불가·안전한 token 매핑 불가 | 0, 이유 보존 | 0, 이유 보존 |
| 원문 자체 누락·hash 변조 | group integrity 실패 | group integrity 실패 |

UNKNOWN 점수는 실제 unknown 원인·선행 의존성이 일치할 때만 준다. 모든 것에 UNKNOWN을 붙이면 정답이 되는 정책이 아니다. 정상 ref에 대한 의미상 거짓 accusation은 REFUTED/-1, 해석 불가나 잘못된 ref는 각각 미검증/구조 오류로 남긴다. 오류 목록에 없는 origin이라는 이유로 제출 자체를 버리지 않는다.

알 수 없는 ref/중복 key/중복 origin은 구조 오류다. raw 보존 후 note audit 실패로 기록하며 해당 unit에 모델이 쓰지 않은 token을 만들어 벌점을 붙이지 않는다. 이러한 후보도 원래 n에는 남는다. 오직 원문·logprob 자체가 유실된 경우에는 그룹 전체를 학습하지 않는다.

```text
A(i,u) = r(i,u) - sum_j r(j,u)/n
```

평균은 MIS 전에 원래 n=2 또는4 전체로 계산한다. 생존 후보만 다시 중심화하지 않는다. 표준편차로 나누지 않는 기존 프로젝트 변형이다. 모두 +1, 모두0, 모두-1인 같은 단위는 상대 신호가0이다.

**점수0과 기울기0은 다르다.** PERSISTED/UNKNOWN의 온라인 점수는0이지만 다른 후보가+1이면 해당 후보의 중심화 계수는 음수가 될 수 있다. 원래 n과 평균에 계속 포함한다. 원문에서 안전하게 대응한 제출 토큰이 있는 경우 그 계수도 적용하고, 미제출은 매핑할 토큰이 없어 기여0이다. 이 동작을 감추기 위해 후보를 버리거나 보상을 재중심화하지 않는다.

### D08.5 목적함수

기존 clipped surrogate를 `S(ρ,A)=min(ρA,clip(ρ,0.8,1.28)A)`로 쓴다. `ρ=exp(current_lp-old_lp)`이며 ref는 KL에만 사용한다. K_i는 기존 MIS가 해당 원래 행동을 유지했는지를 뜻한다. `T(i,u)`는 검증된 원래 token 위치의 집합이다.

```text
L_unit(g,c) = - λ_c /(n × U)
                × Σ_i K_i Σ_u mean_{t∈T(i,u)} S(ρ_i,t, A(i,u))

λ_online = .0625
λ_final  = .0625
```

T가 비어 있으면 그 단위의 합은0이고 분모 U/n은 유지한다. U=0이면 no-unit 상태로0, 성능·노트 성공으로 쓰지 않는다. 원래 group별 가중치 T1/T2/T3=1, DIAG=.5와 usable group의 고정 합 정규화는 유지한다.

기존 `group_loss`의 `n × completion_cap` 나눗셈을 재사용하려면 원래 token 계수를 다음처럼 만든다.

```text
partial_adv[i,t] += completion_cap × λ_c × A(i,u) /(U × |T(i,u)|)
```

그러면 cap이 정확히 한 번 상쇄된다. T2/T3에는 실제 bound cap1024, DIAG에는8192를 쓴다. V58의8192를 모든 길이에 복사하지 않는다. 이 식과 직접 unit 평균 계산이 fixture에서 동일해야 한다. 성분별 clipping은 별도로 계산한 뒤 더하여, answer와 다른 부호의 repair를 먼저 합쳐 비선형 min에 넣지 않는다.

전체 학습은 기존 answer-PG + 위 두 unit-PG + 기존 KL이다. 새 프로토콜의 과거 strict/fact/assessment PG는0이다. legacy protocol의 목적함수는 그대로 복원한다. KL은 기존 k3 형태 `expm1(ref-current)-(ref-current)`와 β=.01을 사용하고 같은 행동에 두 번 부과하지 않는다. 각 component 재호출 시 β=0으로 하거나 기존 같은-forward 분리 식을 사용한다.

실제 `group_loss()` 호출과 대응하는 결합식의 정본은 다음과 같다. `M`은 원래 전체 action mask, `K`는 같은 원래 행동의 MIS 유지 값, `C_g`는 bound completion cap이다.

```text
w_g = group_weight_g / sum_usable_groups(group_weight)
c_answer[i,t] = A_answer[i] * solution_token_mask[i,t]
c_c[i,t] = C_g * lambda_c/U_g *
             sum_u A_c[i,u] * 1[t in T(i,u)] / |T(i,u)|

phi(r,c) = min(r*c, clip(r,0.8,1.28)*c)

L = sum_g w_g/(n_g*C_g) * sum_i K_i * sum_t M[i,t] *
      (-phi(r,c_answer) - phi(r,c_online) - phi(r,c_note) + .01*k3)
```

답 보상 벡터에는 cap 상쇄를 적용하지 않는다. 기존 함수의 실제 인자명에 맞춘 연결 예시는 다음과 같다. `zeros`는 batch 크기의0 벡터, `M`은 전체 action mask, `K`는 기존 seq_keep다. 코드에 그대로 복사할 완성 함수가 아니라 caller의 대응 계약이다.

```python
L_c = group_loss(cur_lp, old_lp, ref_lp, zeros, M, K,
                 eps_low=.2, eps_high=.28, beta=0,
                 norm_const=n_g*C_g, partial_adv=c_component)
L_KL = group_loss(cur_lp, old_lp, ref_lp, zeros, M, K,
                  eps_low=.2, eps_high=.28, beta=.01,
                  norm_const=n_g*C_g, partial_adv=None)
```

answer/online/final 각 성분을 따로 계산한 뒤 KL을 한 번만 더한다. 벡터를 먼저 더해 한 번 clipping하는 구현은 다른 목적함수이므로 금지한다. usable group 정규화는 모든 채널의 사전 유효성으로 한 번 결정하며, 채널마다 group 집합을 바꾸지 않는다.

### D08.6 CORRECTED 전용 기여 증명

현재 assessment 합산 패턴을 재사용하여 다음을 측정한다.

1. 원래 공개 origin과 실제 모델이 낸 CORRECTED 주장의 raw token을 결합한다.
2. 검증 통과한 양의 후보와 같은 단위의 음의 후보를 모두 포함한다.
3. MIS 뒤 유지된 후보의 **CORRECTED 채널만** VJP를 구한다.
4. 파라미터 identity별로 부호를 보존하여 합산한다. norm의 합을 대신 쓰지 않는다.
5. answer·KL·PERSISTED·OUTPUT·fact·I/U classification 기여를 제외한다.
6. 같은 원래 계수에서 **양의 VERIFIED native CORRECTED 항만** 따로 뽑은 VJP도 측정한다. 이 양성 부분과 전체 signed aggregate가 모두 유한·비영이어야 “검증된 수정 항의 기여”를 인증한다.
7. retained positive verified correction≥1과 실제 optimizer 변경·저장·fresh reload를 연결한다.

native CORRECTED 감사 채널은 실제 native 수정 witness에 연결된 온라인 주장/patch와 최종노트의 CORRECTED 주장만 대상으로 한다. 반증된 CORRECTED 주장에 붙은 음의 계수도 signed 합산에서 제외하지 않는다. PERSISTED·UNKNOWN·형제 비교는 전체 학습에 원래 계수대로 남지만 이 전용 감사 마스크에서는 제외한다. 이 마스크를 적용하면서 평균이나 분모를 다시 계산하지 않는다. 온라인 채널/최종노트 채널도 따로 보고하며 한쪽 성공으로 다른 쪽의 기여를 인증하지 않는다.

개별 norm은 양수여도 서로 상쇄될 수 있다. 전체 loss 숫자가 작다고 기울기가0인 것도 아니다. 이 검사만의 VJP를 계산한다고 normal `.grad`를 더하거나 optimizer를 두 번 호출하면 안 된다. 한 forward의 graph에서 보조 미분을 관측하고 원래 backward와 결과 일치를 테스트한다.

그 결과는 “수정 관련 보상이 업데이트에 기여했다”는 기능 증거다. 그 기여만의 최종 AdamW 파라미터 효과나 성능 인과효과는 별도 대조 없이는 주장하지 않는다.

### D08.7 무신호 처리

| 이유 | 기록 | 후속 |
|---|---|---|
| 모든 후보가 똑같이 맞음 | UNIFORM_VERIFIED_REWARD | 해당 unit 상대 신호0, 실패를 만들지 않음 |
| 모두 틀리거나 미제출 | NO_VERIFIED_CORRECTION | 노트 학습 성공 아님, 실제 원문으로 원인 분석 |
| MIS가 전부 제외 | NO_RETAINED_REPAIR_SIGNAL | sampler/training 입력·정책 차이 검사 |
| 합산 기울기0 | ZERO_AGGREGATE_REPAIR_GRADIENT | cancellation/마스크/정규화 점검 |
| 전체 비영·검증된 양성 부분0 | NO_VERIFIED_POSITIVE_REPAIR_GRADIENT | 거짓 주장 억제만 있었는지 구분, S 미통과 |
| 정답 채널만 유효 | ANSWER_ONLY_UPDATE | 기능 완료 가능, 수정 의미 미검증 |
| 노트 분류만 유효 | CLASSIFICATION_ONLY | V58형 성공 범위, native repair와 구별 |

고정된 최소시험에서 신호가 없다는 이유만으로 표본을 정답이 잘 나오는 문제로 갈아끼우지 않는다. 추가 표본·seed·프롬프트 실험이 필요하면 사전 고정한 별도 protocol로 기록하고 이전 실패를 유지한다.

<a id="D09"></a>
## D09. Vision / Merger / LLM 학습 경로

실제 연결부는 마지막 Merger 한 개와 DeepStack Merger 세 개다. `modules.inspect_modules`로 이름과 shape를 확인하고 Merger를 Vision과 중복 집계하지 않는다.

| 그룹 | 실제 학습 대상 | 현재 설치된 tensor/원소 |
|---|---|---|
| L | 언어 Decoder의 LoRA | 504 / 33,030,144 |
| M | 네 Merger의 FP32 trainable copies | 24 / 109,105,152 |
| V | Vision의 LoRA | 96 / 2,359,296 |
| base | NF4 기저·기타 frozen weights | 동결 |

LoRA rank16/alpha32/dropout0/bias none을 유지한다. 실제 변경 텐서 수는 설치 수보다 작을 수 있다. 이름에 q_proj가 있을 것이라고 가정하지 말고 Vision qkv/proj와 언어 q_proj/k_proj/v_proj/o_proj를 실제 inventory로 구분한다.

```text
independent solution audit
  known wrong required observation / proved false visual relation → VISION_ON
  every required observation verified, no conflict/unknown         → VISION_OFF
  otherwise                                                       → UNRESOLVED
```

- ON: Vision adapter + Mergers + LLM adapter 업데이트 후보.
- OFF: Vision adapter의 gradient 없음, Mergers + LLM adapter 업데이트 후보.
- UNRESOLVED: route admission 없음. wire의 legacy OFF 기본값을 승인으로 읽지 않는다.

OFF에서도 이미 학습된 Vision adapter 값은 forward에 사용한다. “Vision 제거 모델”이 아니다. 이전 ON에서 생긴 momentum이 OFF-only step에 Vision을 바꾸지 않는지 `grad=None`과 optimizer 전후 hash로 검증한다. ON/OFF가 한 batch에 섞인 결과는 OFF-only 동결 증거를 대신하지 않는다.

정확한 관측을 거꾸로 계산했다면 nonvisual 연산 오류 근거가 된다. 그렇다고 내부 LLM만 원인이라고 확정하지 않는다. 이는 **검증 근거에 따른 실험적 업데이트 정책**이며 latent module causal attribution이 아니다.

<a id="D10"></a>
## D10. 코드 변경 지도와 인터페이스 소유권

### D10.1 재사용할 현재 계층

```text
fine_tuning/
  modules.py                              실제 모듈 탐색
  configs/                                고정 protocol / source registry
  notebook_rl/
    schema.py / vault.py                  actor-private-lineage 분리
    dataset.py / local_data.py            데이터 생성·고정·gate
    prompts.py / local_profile.py         문법·protocol 버전
    engines.py / shared_policy_engine.py  입력·이미지·동일 live policy
    rollout.py                           실제 시도 tree
    episode_notebook.py                   전체 이력·라우팅·재검증
    observations.py / answers.py          관측·최종답
    reasoning_trace.py / circuit_trace.py native 연산 검사
    verifier.py / verifier_service.py     비공개 검사 프로세스
    notebook_synthesis.py                의미·ancestry 검사
    synthesis_credit.py / fact_credit.py  기존 원문 token 매핑
    policy.py / quantized_policy.py       NF4·adapter·four Merger
    losses.py / updates.py               PG·KL·MIS·실제 update
    checkpoint.py / trainer.py            저장·동기화
    local_run.py / local_train.py         최소시험·고정10
    local_eval.py / local_ladder_recovery.py  별도 평가 경로
    repair_contract.py                   NEW: 문법/원문 span/typed note
    repair_credit.py                     NEW: actual repair/성분 credit
  tests/                                 기존 회귀 + 새 반례
```

위 트리에서 새 파일 두 개만 신규다. 현재 source의 하위 파일·정확한 함수는 D18 AST 카탈로그가 정본이다. 존재하는 파일명과 구현 예정 이름을 혼동하지 않는다.

### D10.2 변경 계약 표

| 소유 모듈 | 기존 함수/접점 | 수정 책임 | 그대로 둘 것 |
|---|---|---|---|
| repair_contract NEW | D06 API | strict grammar, 공개 inventory, 원문 span | gold/private checker 접근 금지 |
| prompts/local_profile | build_protocol/restore_protocol/local_prompts | 새 identity·문법·budget·reward 계약 | legacy 복원 hash |
| rollout | build_tree/select_failures/continuation_ids | 검증된 과정 모순 retry, joint response 적용 | 원래 parent token prefix·n·정책 |
| observations/answers/trace | observation_spans/extract_final/check_trace | 새 solution 범위만 검사, 원문 offset lift | legacy 의미 규칙·정답 비교 |
| verifier_service | grade/trace RPC | 새 contract version과 raw+section identity 확인 | actor gold 격리·source guard |
| episode_notebook | audit_episode/public_transcript/append_episode_notebooks | all-history compact note, 새 audit 연결 | 전체 분기 보존·strict_routing |
| notebook_synthesis | _lesson/audit_episode | 기존 native 관계 의미의 작은 wrapper | 구형 완전노트 채점 |
| repair_credit NEW | D06 API | 원래 unit/group 점수·raw token credit | 임의 문장 정답 생성 금지 |
| updates | has_learning_signal/group_has_signal/partition_trees/optimizer_step/aggregate VJP | online-only 진입·성분 mask·새 unit 식·양성/전체 합산 기여 | 실제 old/ref/한 번 update·동결 검사 |
| local_run/local_train | capture/train/admission | F와S 별도 결과·고정 probe | V58 success 보존·guard 약화 금지 |
| local_eval/ladder_recovery | evaluate/diagnose | 같은 새 문법/채점, n1 독립 평가 | 평가 중 optimizer0 |
| checkpoint/cache | save/load 및 binding | protocol metadata 결합·fresh 검증 | tensor 저장 형식·공유 base |
| tests | 해당 소비자 tests | 원문·분모·누설·반례 | 역사 프로토콜 회귀 |

### D10.3 함수별 책임 — 신규 구현자의 최소 계약

#### `parse_retry_sections`

입력은 모델 raw 문자열과 명시적 버전이다. 출력은 정확한 check/solution char span과 raw hash다. 파일 I/O·tokenizer·gold·모델 호출을 하지 않는다. 문자열 안의 marker·CRLF·Unicode·중복 marker·끝이 잘린 JSON을 구별한다. 현재 오류 판단은 하지 않는다. 문법 실패는 typed parse failure로 반환/전달하고 원문을 잃지 않는다.

#### `project_solution` / `lift_solution_spans`

projection은 checker의 임시 view다. Sample.text·completion_ids는 바꾸지 않는다. 기존 checker가 만든 offset을 원문에 올려 동일 substring과 hash를 assert한다. 잘린 span·다른 raw hash·중첩 range 충돌이면 fail closed다. 한 모듈에서 변환 규칙을 정하고 관측·답·trace·reference consumers가 같은 것을 쓴다.

#### `public_origins`

원문·sample_id·parent_id·순서만 읽는다. 현재 `whole_reference_notebook.public_inventory`의 공개 부분을 재사용한다. 정답을 맞힌 항목만 추리거나 private issue 존재로 ref를 만들지 않는다. 같은 의미의 여러 발생 위치를 잃지 않고 occurrence 목록으로 보존한다. 결정적 정렬·digest와 안정적 ref identity가 필요하다.

#### `audit_online_repair`

parent/child가 같은 problem·policy이며 실제 descendant인지 먼저 검사한다. child ErrorCheck 원문과 child solution을 분리하고 양쪽을 fresh audit한다. 원래 operation identity와 실제 수정값·rule 적용이 일치하는지 검사한다. final answer가 틀려도 검증된 부분수정을 기록한다. parent의 정답 여부만으로 전체 과정이 맞다고 추정하지 않는다. 반환에는 private evidence가 있을 수 있으므로 actor prompt generator가 이 객체를 받지 못하게 한다.

#### `audit_final_note`

원래 모든 시도와 지정 성공 ID를 확인한다. 제출된 모든 claim을 순회한다. private 오류 목록에 없는 주장도 확인하여 거짓 accusation을 놓치지 않는다. 참·거짓·unknown·missing·native correction·persisted·output을 따로 계산한다. 최종노트의 correction이 online ErrorCheck와 일치하는지 추적하되, 최종노트가 먼저 수정한 것처럼 기록하지 않는다.

#### `repair_group_credit`

그룹의 원래 n·prompt·policy·parent가 같은지 검증한다. T2/T3n2와 DIAGn4를 모두 처리하지만 legacy synthesis_credit._group의 n4 제약은 바꾸지 않는다. 공개 inventory와 완료 cap을 결합하고 점수→평균→advantage→원문 token span을 계산한다. host-expanded text의 토큰 수를 쓰지 않는다. caller는 모든 raw member를 반드시 전달한다.

#### `optimizer_step`의 새 분기

기존 protocol dispatch에만 새 objective를 추가한다. answer mask와 component vector를 분리하고 clipping·KL 정규화가 기존과 다르지 않은 부분은 그대로 둔다. source·raw 재검증은 forward 전과 commit 직전에 유지한다. 같은-forward VJP 누적은 `.grad`를 오염시키지 않아야 한다. commit 이후 관찰 callback 실패로 이미 된 step이 안 된 것처럼 기록되지 않도록 기존 원칙을 유지한다.

현재 `updates.py`의 `has_learning_signal()`/`group_has_signal()`은 T2/T3에서 scalar 답 보상만 확인하고, `optimizer_step()`은 그 결과로 미리 종료할 수 있다. 새 protocol에서는 답 보상이 전부0이더라도 **resolved route + 독립적으로 검증된 온라인 수정 단위의 상대 신호**가 있으면 진입해야 한다. 최종 성공 답이나 `rewarded_notebook.status==GENERATED`를 online-only 학습의 조건으로 두지 않는다. 두 후보의 답은 모두 틀렸지만 한 후보만 실제 연산을 고친 fixture에서 partition→optimizer 직전까지 연결되는지 검사한다. 반대로 unknown claim만 있거나 실제 토큰/상대 신호가 없으면 진입시키지 않는다.

`local_profile.restore_protocol()`은 현재 상수로 과거 protocol을 재구축해 동일성을 비교한다. 새 prompt/검증기 상수를 전역 교체하면 기존 run의 복원까지 깨질 수 있다. 버전별 정확한 builder를 선택하고 옛 prompt·hash·보상 식을 byte-equivalent하게 복원하는 회귀를 추가한다. 새 protocol ID를 허용 목록에 넣는 것만으로는 호환성 보존이 아니다.

#### 최소시험과 고정10 제어

`local_run.capture`는 F·S 증거를 별도 필드로 저장한다. S가 실패해도 F가 성공한 기록은 지우지 않는다. `local_train.train`은 기존 고정 입력·초기값·한도와 새 protocol을 결합한다. 최소 성공 가중치를 몰래 이어서 freshF0 대조군을 시작하지 않는다. 각 arm의 초기 trainable hash를 비교한다.

#### 평가 경로

`local_ladder_recovery`는 build_tree를 그대로 호출하지 않는 별도 경로다. 새 grammar·feedback·projection·note score를 여기도 적용해야 한다. 학습용 group n≥2를 위해 평가 응답을 복제하지 않는다. 평가 n1은 학습 usable=false다. 평가 상태에서 optimizer 호출이 일어나면 hard failure다.

### D10.4 UML 성격의 관계

```text
ActorItem ──uses──> RolloutRequest ──prepare──> InputPreparation
PolicyBundle ──owns──> NF4Base[frozen] + L/M/V[trainable inventory]
SharedPolicyRolloutEngine ──references──> PolicyBundle (동일 객체)
RolloutRequest ──generates──> Completion ──records──> Sample
Group 1 ──contains n──> Sample; ProblemTree 1 ──contains──> Group
Sample ──parses──> RetrySections ──projects──> SolutionView
VerifierService ──privately uses──> VerifierItem / native world
SolutionView + private world ──checks──> TraceReport / SlotReport
parent Sample + child Sample + reports ──checks──> RepairWitness
raw model note + public origins + reports ──checks──> NoteAudit
Group + audits + original token map ──creates──> ComponentCredit
ComponentCredit + old/ref/current LP ──differentiates──> signed VJP + gradient
optimizer ──commits──> trainable tensors ──persists──> checkpoint
fresh process ──loads──> standalone NF4 + checkpoint ──verifies──> reload evidence
```

<a id="D11"></a>
## D11. 하드웨어·메모리·저장·폴더 이식성

### D11.1 8GB에서 실제로 큰 부분

총 trainable 원소는144,494,592다. FP32 파라미터·gradient·Adam의 두 moment만 대략 `×16 bytes`로 약2.153GiB다. 여기에 frozen NF4 base, 비양자화 부분, image activation, 긴 history의 forward/backward, KV, allocator가 더해진다. 합산 감사의 CPU FP64 accumulator만 약1.077GiB다. 이 산술은 실제 peak 측정의 대체가 아니다.

이전 실행의 allocated/reserved가 물리8GB보다 컸다고 “GPU VRAM에 전부 들어갔다”고 쓰면 안 된다. SMI 사용량, allocator peak, CPU RSS, driver/shared memory 가능성을 분리한다. CPU 100%나 GPU util만으로 병목 위치를 단정하지 않는다.

기존 Torch2.8.0+cu128 / Transformers4.57.1 / PEFT0.17.1 / Accelerate1.11.0 / bitsandbytes0.48.2 pins를 먼저 확인한다. 네 Merger와 lm_head의 quantization 제외 및 실제 trainable 수를 검사한다. 라이브러리 업그레이드는 이 설계의 필수 작업이 아니다. [Transformers4.57.1 문서](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes), [PEFT0.17.1 준비 함수](https://raw.githubusercontent.com/huggingface/peft/v0.17.1/src/peft/utils/other.py)

### D11.2 실패 시 정책

- OOM이면 실제 한도·단계·입력 길이·메모리를 저장하고 중단한다.
- 모델을 더 작게 바꾸거나 Vision을 빼고 원래 계획 성공이라고 하지 않는다.
- 이미지 해상도·context·출력 cap·optimizer를 자동 변경하지 않는다.
- 다른 조건이 필요하면 새로운 protocol로 분리하여 양 arm에 같이 적용한다.
- 무작위 tiny model은 mask·손실·저장 함수 fixture용이다. 실제 자기수정 성공으로 보고하지 않는다.
- 모든 checkpoint는 새 output root에 쓰며 완료 marker와 해시가 닫힌 뒤에만 사용할 수 있다.

### D11.3 저장 묶음

| 묶음 | 필요한 내용 |
|---|---|
| standalone base | NF4 weights·quantization state·config·processor·tokenizer·원출처/revision |
| trainable | adapter config/state·네 Merger trainable payload·inventory·tensor hashes |
| runtime | optimizer·scheduler·step·cursor·RNG·protocol/source/data bindings |
| evidence | raw responses·sampler LP·원문 token·reward·mask·VJP·actual changes·sync probes |

새 프로세스는 원본 BF16 경로에 접근할 수 없는 환경에서 저장묶음만으로 실제 단일/복수 이미지 forward·짧은 생성·optimizer state를 복원해야 한다. 단순 torch.load 성공은 부족하다. 이전 shared BF16 두 blob은 기존 checkpoint 여섯 개의 재현 의존성이 있으므로 삭제 적격으로 간주하지 않는다. 이번 산출물 작성에서는 캐시를 삭제하지 않는다.

### D11.4 fine_tuning 단독 이식

새 구현이 import할 수 있는 프로젝트 코드는 fine_tuning 안으로 한정한다. MMDL/src/mmdl/experiments 또는 개발자 홈 절대경로에 런타임 의존성을 추가하지 않는다. 데이터·모델·output root는 명시적 CLI/환경 인자로 받는다.

portable 검증은 실제 fine_tuning만 복사한 임시 root에서 실행한다. Python cwd·sys.path가 원래 checkout으로 돌아가지 않는지 child 프로세스까지 확인한다. 원본 소스 접근을 막았을 때 import·CPU gate·실제 offline reload가 동작해야 한다. 과거 source-bound 승인은 새 경로에서 hash/상대경로 identity로 재검증하며 절대경로를 무조건 무시하지 않는다.

<a id="D12"></a>
## D12. 실험 순서와 성능 평가

### Gate 순서

| Gate | 실행 | 통과하면 |
|---|---|---|
| G0 | 요구·schema·수식·source·소유권 고정 | CPU 구현 |
| G1 | 원문 경계·offset/token·private 분리 | verifier/rollout 연결 |
| G2 | 의미 검증·전체 이력·독립 routing | CPU 목적함수 통합 |
| G3 | 원래 group 중심화·손검산·분리 mask·부호 상쇄 | source-bound 준비 |
| G4 | 데이터·source/cache/protocol·실제 입력 용량 | 사전 고정한 bounded probe |
| G5 | 실제 보존 신호·F/S 구분·양성/전체 VJP·업데이트 | CPU 재생·실물 저장 검사 |
| G6 | 소유 종료·실물624·새 PID 복원·NF4 불변 | 고정10 engineering trial 판단 |
| G7 | 고정10·같은 별도 evaluator·R0/R1·불확실성 | 측정된 범위의 효과 보고 |
| REMOTE-BF16 | BF16 원격 소규모 재검증 | 별도 사용자 승인 필요 |
| LARGE | 대규모 학습/전체 MMMU·Pro | 별도 승인과 평가 조건 동결 |

Gate 번호는 `plan.md`와 같다. F/S/E는 Gate 번호가 아니라 서로 다른 증거 수준이다. F가 이미 V58에서 된 부분을 새 발견으로 반복하지 않는다. 새 문법·새 correction credit가 변경하는 부분만 실제 연결 증거가 추가로 필요하다. G5에서 S 무신호이면 이를 숨기지 않고 어느 단계가 막혔는지 닫는다. 양경로의 ON/OFF-only 변경·동결은 G5/G6에서 각각 검증하며 한 경로만 실제 실행되면 나머지는 미검증이다. F만 성공한 경우 그 증거를 보존하지만, S나 효과를 자동 인증하거나 대규모 학습으로 넘어가지 않는다.

### 대조군

| arm | 생성·노트 | answer PG | repair/note PG | KL |
|---|---|---:|---:|---:|
| F0 | 학습 전 동일 NF4, 평가만 | 없음 | 없음 | 업데이트 없음 |
| R0 | 동일 ErrorCheck·최종노트 경로 | 예 | 0 | 동일 |
| R1 | 동일 ErrorCheck·최종노트 경로 | 예 | 예 | 동일 |

R0도 노트를 생성하므로 “노트 자체가 없는 대조군”이 아니다. 노트/수정 보상 제거의 절차 비교다. 학습 후 정책이 달라지면 실제 토큰·정답·유효 그룹·update 수가 달라질 수 있다. 같은 seed·상한만으로 compute matched라고 하지 않는다. 실제 노출·분모·업데이트·시간을 보고한다.

추가적으로 고정 F0/FT에서 ErrorCheck를 원래대로/제거/다른 문항의 것으로 바꾼 작은 inference 실험을 할 수 있다. 이것은 텍스트 조건 의존성을 보며 내부 모듈의 유일한 원인을 증명하지 않는다. 오염·정답 노출 없는 고정 실험으로 별도 등록한다.

### 보고 지표

| 지표 | 분모·정의 |
|---|---|
| first_attempt_accuracy | 고정 평가 문항, 문항당 사전 지정 첫 sample 한 개 |
| retry_recovery_rate | 그 첫 시도 오답 문항 중 고정 횟수 내 해당 계보에서 정답 |
| correct_to_wrong | 처음 정답이었으나 검토 뒤 오답으로 변한 비율 |
| verified_native_repairs | 독립 확인한 operation correction 수·문항 수 따로 |
| introduced_process_errors | 수정 뒤 새 VIOLATED operation/observation |
| note_precision / coverage | D07, 빈 분모N/A |
| ON/OFF/UNRESOLVED | 독립 routing 상태별 수·학습 여부 |
| no_signal / MIS_rejected | 검증 보상·실제 보존 신호 구분 |
| 비용 | 입력/이미지/출력 토큰, 추가 note 호출, 생성/backward/검사/복원시간 |
| 메모리 | allocated/reserved/SMI/RSS 실제 측정별 값 |

seed3407의 첫 시험 뒤 반복이 필요하면3408/3409를 사전 추가한다. 반복 seed를 독립 문항 수처럼 세지 않는다. shared image·native topology·source family를 묶은 paired 비교/cluster bootstrap을 사용한다. 10개 개발 문항의 통계적 불확실성은 크게 남는다.

### 62%를 어떻게 넘어섰다고 검증할 것인가

현재 558/900·62.00%는 과거 BF16·32K·V8 조건이다. 로컬 NF4·1024 기능시험과 직접 점수 비교하지 않는다. 차후 사용자 승인 뒤 원격 BF16에서 **동일 새8K 평가 프로토콜의 미학습 baseline과 학습 모델**을 같이 평가한다. 모델·데이터 revision, 이미지 처리, prompt, sampling, V8 파서, 전체900 분모를 맞춘다. 과거62%는 참고 행으로만 둔다.

MMMU/Pro 평가 오답을 다시 train으로 보내지 않는다. 개발 중 validation을 반복 사용해 선택했다면 그 사실을 보고한다. 개선이 관측되어도 초기 작은 합성 그래프의 성공만으로 모든 전문 실패군에 일반화되었다고 말하지 않는다.

<a id="D13"></a>
## D13. 테스트 설계와 필수 반례

새 tests는 `fine_tuning/tests/`에 둔다. 아래는 **구현할 검사 목록**이며 이 문서 작성 중 통과시킨 모델 시험이 아니다.

| ID | 검사 | 기대 |
|---|---|---|
| T01 | gold/정답 trace/expected issue에 actor loader 접근 | 접근 차단 |
| T02 | benchmark-derived parent → 합성 child | training_allowed=false 전파 |
| T03 | 검출 중복 후보 미검토 | 실행 중단 |
| T04 | 같은 seed 다른source를 독립source라고 표기 | 거부/별도 범위 표시 |
| T05 | ErrorCheck 안 과거 Final answer | 새 답으로 추출하지 않음 |
| T06 | ErrorCheck 안 과거 Trace/Observed facts | 새 trace·관측에서 제외 |
| T07 | JSON 안 marker/CRLF/Unicode | 원문 span 정확 |
| T08 | 중복 marker/중복key/NaN/bool숫자 | 계약 거부 |
| T09 | projection 후 lifted span substring/hash | raw와 동일 |
| T10 | actual completion token decode·span 경계 불안정 | credit 금지, 기록 |
| T11 | parent 아닌 형제의 정답 | CORRECTED 불가 |
| T12 | 같은 index 다른 node/operands | 동일 operation 불가 |
| T13 | 관측 수정했지만 적용은 역방향 | 두 단계 판정 분리 |
| T14 | 최종답 맞지만 관측 VIOLATED | 답CORRECT+processcontradicted, retry 가능 |
| T15 | 정답인데 trace 없음 | UNKNOWN, 정상 과정 인증 금지 |
| T16 | 부분 operation 수정·최종답 여전히 오답 | native repair 확인 가능 |
| T17 | PERSISTED/OUTPUT만 성공 | nativeCORRECTED gate 실패 |
| T18 | 정상 ref를 오류라고 주장 | false accusation 보존·벌점/실패 |
| T19 | 빈 claim/누락 attempt/성공ID 교체 | 완성 노트 성공 금지 |
| T20 | 미확정 의존성 묵살 | claim 미검증·coverage 누락 |
| T21 | n2 online을 legacy n4에 억지 복제 | 거부 |
| T22 | 원래 rewards[1,0] | advantages[.5,-.5] |
| T23 | [1,1],[0,0],[-1,-1] | 전부 상대 신호0 |
| T24 | MIS로한후보제외 | 원래평균유지, 재중심화 금지 |
| T25 | answer PG가 ErrorCheck/DIAG/history에 있음 | 검사 실패 |
| T26 | unit 식과 cap상쇄 token식 | 손검산 동일 |
| T27 | token 두 unit 중복 | double credit 거부 |
| T28 | 개별nativeVJP양수·합산상쇄 | native gate 실패 |
| T29 | classification VJP만 비영 | 수정 의미 미검증 |
| T30 | auxiliary VJP 전후 기존.grad | 의도한 계산 외 불변 |
| T31 | ON→OFF-only, optimizer momentum잔존 | V.grad=None·V값불변 |
| T32 | mixed ON/OFF update | OFF-only검증으로표기금지 |
| T33 | fresh 모델base/quant state 변경 | commit거부 |
| T34 | source/raw/verifier 변경 afterforward | commit전재검사거부 |
| T35 | 저장중중단/marker없음 | reload불가 |
| T36 | 새 process원본BF16접근 | 차단·audit |
| T37 | 단일/복수imagegrid/pixel변경 | 재로딩동일성실패 |
| T38 | fine_tuning단독복사/import원래repo | 차단 |
| T39 | n1evaluation에optimizer | hardfailure |
| T40 | context/OOM/budgetstop | raw보존·자동한도완화0 |
| T41 | 구protocol restore·replay | 이전결과/hash동일 |
| T42 | 고정평가출력의sample0대신best선택 | 지표계산거부 |
| T43 | 모든 답0·DIAG없음·online 수정 분산 있음 | resolved route/token 검증 후 optimizer 진입 |
| T44 | verifier false이나 미검증/구조오류/점검 의도 | REFUTED와 분리·자동 -1 금지 |
| T45 | 정상 public origin에 거짓 CORRECTED 비난 | 생성 전 U에 유지·REFUTED/-1 |
| T46 | signed VJP 비영·verified-positive VJP0 | S 미통과·F 증거는 보존 |
| T47 | zero reward와 다른 후보+1 | 원래 n 중심화 후 음의 계수 가능·분모 유지 |
| T48 | 전역 상수 교체로 old restore 달라짐 | exact builder 회귀 실패·역사 run 보호 |

### 기존 테스트 재사용 지점

- `test_notebook_rl_schema.py`, `test_notebook_rl_vault.py`, `test_notebook_rl_verifier_service.py`: 경계·자료형.
- `test_notebook_rl_reasoning_trace.py`, `test_notebook_rl_observations.py`: 과거 주장·의존성·연산 의미.
- `test_notebook_rl_synthesis_profile.py`, `test_notebook_rl_synthesis_credit_updates.py`, `test_notebook_rl_synthesis_credit_admission.py`: 버전·credit·실제 update 경로.
- `test_notebook_rl_policy.py`: Vision gradient·momentum·동결.
- `test_notebook_rl_whole_reference_notebook.py`, `test_notebook_rl_whole_reference_credit.py`: 공개 refs·raw mapping.
- `local_eval`/`local_ladder_recovery` 관련 실제 tests는 D18 목록에서 찾아 caller와 함께 검사한다.

일부 이름·위치는 checkout 변경에 따라 달라질 수 있으므로 `rg --files fine_tuning/tests`로 존재를 확인한다. 새 `test_notebook_rl_repair_contract.py`, `test_notebook_rl_repair_credit.py`는 아직 없는 **구현 예정 파일**이다.

```bash
# 구현 후 실제 파일 존재를 확인한 뒤 쓰는 집중 검사의 형태
python -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_repair_*.py'
# 기존 suite는 영향 범위가 결합된 최종 단계에서 실행
python -m unittest discover -s fine_tuning/tests -p 'test_notebook_rl_*.py'
```

환경별 private cohort skip을 PASS로 숨기지 않는다. Ruff/mypy 범위·옵션과 외부 import 미검사 범위를 보고한다. CPU fixture가 통과해도 실제 Qwen 생성·가중치 변경 시험과 별도로 적는다.

<a id="D14"></a>
## D14. 기존 코드에서 안전하게 이행하는 순서

1. 현재 branch/status·핵심 source SHA·MMDL 상태와 보호 파일을 기록한다. 기존 사용자 변경을 reset하지 않는다.
2. 새 문법·자료형을 순수 함수로 구현한다. 기존 모델·runtime을 아직 바꾸지 않는다.
3. projection을 모든 checker/RPC/inventory/replay에 연결한다. prompt만 먼저 바꾸지 않는다.
4. 동일 parent의 실제 repair semantics와 token credit를 CPU 반례로 검증한다.
5. 새 protocol 분기에서만 online ErrorCheck·compact final note·새 objective를 활성화한다.
6. updater의 마스크·성분 clipping·합산 VJP와 admission을 통합한다.
7. 별도 평가 caller와 folder portability를 연결한다.
8. 실제 tokenizer CPU 입력검사 후 사전 고정한 1–2문항만 실제 실행한다.
9. 실제 update가 있다면 종료 원문 감사·실물 checkpoint·새 프로세스 복원을 닫는다.
10. 기능/의미 결과에 따라 고정10·두 경로·대조군을 진행한다. 무신호를 성공으로 바꾸지 않는다.

현재 버전 수가 많다고 기존 코드·기록을 일괄 삭제하지 않는다. 새 생산 경로는 하나로 정하되 legacy restoration은 유지한다. 후보와 채택을 혼동하지 않게 manifest에 protocol_id·source_sha·objective_version·note_contract·response_contract를 함께 저장한다.

### 중단과 재개

현재 checkpoint의 cursor가 있다는 것만으로 arbitrary resume가 지원되는 것은 아니다. raw group 중간에 멈추면 마지막 완료 group과 완료 update를 구분한다. 재개는 기존 코드가 검증하는 범위만 사용한다. 지원하지 않는 중간 상태는 재실행을 새 run으로 기록하고 원문을 덮어쓰지 않는다.

프로세스 ID만으로 실행을 식별하지 않는다. 시작 tick/cwd/cmd/output root를 함께 확인한다. 사용자 pause/종료 요청을 보존하고 “상태가 어때?”를 자동 재개로 해석하지 않는다. 실험 종료시 소유 PID와 GPU 사용 종료를 확인한다. 이 문서 작성은 GPU를 시작하지 않는다.

<a id="D15"></a>
## D15. 남는 위험과 해결을 주장하면 안 되는 부분

| 위험 | 지금의 대응 | 남는 한계 |
|---|---|---|
| 정확한 native repair가 생성되지 않음 | 실제 후보·검증·무신호 기록 | 보상만 설계해도 탐색이 생긴다는 보장 없음 |
| compact claim이 과거 방식 반복 | 온라인 점검→실제 수정의 추가 연결 | 모델이 지시를 지킬지는 실측 필요 |
| 정답인데 잘못된 과정 | 별도 process audit·bounded retry | 답 보상이 solution 전체 의미를 보증하지 않음 |
| 설명을 그럴듯하게 꾸밈 | 닫힌 witness만 검증·학습 | free lesson의 내부 충실성 불가 |
| 원인별 모듈 라우팅 | 관측/연산의 독립 근거 | Vision/Projector/LLM의 유일 원인 판정 아님 |
| SFT 없이 보상 희소 | 실제 부분수정 단위 보상·그룹 | 모두 같은 결과면 상대 신호0 |
| reward hacking | 고정분모·거짓주장·중복·진행보너스0 | verifier 미지원 의미를 악용할 여지 |
| 첫 시도 악화·수정 보상 증가 | 첫 시도 정답률과 실제 오류 수를 회복률과 함께 비교 | 미래 보상을 첫 시도로 전파하지 않아도 공유 파라미터의 간접 영향은 남음; 회복률 상승만으로 개선 선언 금지 |
| 합성→MMMU 전이 | 이후 출처·형태 분리 평가 | 소규모 그래프 성공은 전이 증거 아님 |
| 8GB | 실제 메모리·시간·OOM기록 | 긴 전체이력 optimizer 적합성 보장 없음 |
| 라벨/렌더 오류 | solver/renderer/AI·사람 검토 구분 | 프로그램 truth와 실제 픽셀 불일치 가능 |
| 미래 구현자 drift | 고정계약·test·작업경계·handoff | 문서만으로 무오류 구현 보장 불가 |

특히 “62점을 넘기는 방법”은 설계만으로 확정할 수 없다. 이 구조는 **무엇을 학습했는지 확인할 수 있고, 그 학습이 새 문제에 도움이 되었는지 공정하게 검사할 수 있게 하는 방법**이다. 성능을 반드시 올려야 한다는 목표 때문에 gold 누설·오답 구제·평가 표본 교체를 허용하지 않는다.

<a id="D16"></a>
## D16. 작은 작업 단위로 구현하기 위한 규칙

### 역량 등급

| 등급 | 작업 | 필수 검토 |
|---|---|---|
| L1 bounded | 순수 parser fixture·자료형·문서·기존함수의 작은 연결 | L2 통합 review |
| L2 integration | prompt/profile/RPC/rollout/eval 연결, 다중 caller 수정 | L3 경계 review |
| L3 numerical/architecture | 보상식·마스크·signedVJP·라우팅·누설·checkpoint 의미 | 독립 동급 review + root 승인 |

사용자가 말한 Haiku/Sonnet/Sol/Opus 등의 이름은 미래 작업 배분 예시다. 특정 이름만으로 역량·도구·문맥 크기를 보장하지 않는다. 현재 세션의 실제 위임 정책은 Sol6.1/Astra xhigh다. 계획서는 L1/L2/L3 계약으로 기술하여 다른 환경에서도 그대로 검증할 수 있게 한다.

### 각 작업이 넘겨야 하는 것

```text
task_id / requirement_ids / report_section
base_source_sha / owned_files / read_only_dependencies
function_signatures / input_schema / output_schema
preconditions / postconditions / failure_codes
preserved_invariants / legacy_behavior
executed_checks + actual_exit + logs
fixture_vs_actual_model distinction
changed_files + unresolved_questions
next_task_gate / status
```

하위 작업자가 설계 변경을 발견하면 자기 파일 안에서 임의 호환 옵션을 만들지 않는다. 영향 caller·반례·최소 수정안을 root로 반환한다. 공유 파일은 직렬 통합한다. 테스트는 현재 구현을 그대로 따라한 예제만 쓰지 말고 D13의 잘못된 경우를 반드시 포함한다.

### 상태 정의

DESIGNED=계약만 존재. IMPLEMENTED=실제 호출까지 코드 연결. TESTED=명시된 CPU/static 검사 통과. FUNCTIONAL_VERIFIED=실제 모델update/reload. REPAIR_VERIFIED=실제 native correction 학습 기여. EFFECT_MEASURED=새 평가 완료, 개선 여부와 무관. BLOCKED=구체적 입력/환경/의미 신호가 부족하여 다음 gate 못 넘음. 문서 작성만으로 다른 상태를 붙이지 않는다.

<a id="D17"></a>
## D17. 출처와 인용용 메모

논문 요약과 우리 제안을 분리한다. 인용은 해당 방법의 구성 요소에 한정하고 우리 파이프라인 전체의 검증 근거로 사용하지 않는다. 저장소 main은 바뀔 수 있으므로 실제 구현에 차용할 경우 사용 commit·라이선스·수정 함수를 다시 고정한다.

| ID | 논문·공식 코드 | 인용할 내용 | 직접 사용하지 않는 부분 |
|---|---|---|---|
| S01 | [Training Language Models to Self-Correct via RL](https://arxiv.org/html/2409.12917v2) | 온라인 자기수정·시도별 credit | 모델/과제 성능을 우리 결과로 이전 |
| S02 | [DeepSeekMath](https://arxiv.org/html/2402.03300v3) | 그룹 상대 RL·old/reference | 완전히 동일 GRPO라는 명칭 |
| S03 | [Reflexion](https://arxiv.org/abs/2303.11366), [저자 코드](https://github.com/noahshinn/reflexion) | 반성 메모리와 다음 행동 | 이를 가중치 학습이라고 표현 |
| S04 | [Self-Refine](https://arxiv.org/abs/2303.17651), [저자 코드](https://github.com/madaan/self-refine) | 피드백→수정 순서 | 학습 효과 주장 |
| S05 | [VRRL](https://arxiv.org/html/2607.02490v1), [저자 코드](https://github.com/fc2869/VRRL) | 시각 근거 재검토·실패 prefix | Multi-SFT checkpoint/teacher궤적 |
| S06 | [VL-Rethinker](https://arxiv.org/html/2504.08837v3), [저자 코드](https://github.com/TIGER-AI-Lab/VL-Rethinker) | 강제 재검토 탐색 | 성공궤적 SFT항 |
| S07 | [Vision-SR1](https://arxiv.org/html/2508.19652v1), [저자 코드](https://github.com/zli12321/Vision-SR1) | 지각·정답 보상 구별 | cold-startSFT/미검증Qwen3경로 |
| S08 | [S²R](https://arxiv.org/html/2502.12853v1), [코드](https://github.com/NineAbyss/S2R) | 수정 단계 연구 비교 | SFT초기화·교사 검증자료 |
| S09 | [RISE](https://arxiv.org/html/2407.18219v2), [코드](https://github.com/cmu-mind/RISE) | 다회 자기개선 연구 비교 | reward-weighted지도손실 |
| S10 | [STaR](https://arxiv.org/abs/2203.14465), [코드](https://github.com/ezelikman/STaR) | 자기생성 자료와학습 구분 | 성공풀이 모방/gold rationalization |
| S11 | [R3V](https://arxiv.org/html/2411.00855v1), [코드](https://github.com/njucckevin/MM-Self-Improve) | 멀티모달 자기수정 연구 비교 | filtered출력NLL |
| S12 | [Lessons of Developing PRMs](https://arxiv.org/html/2501.07301v1) | 결과정답과단계정확성 분리 | 새PRM지도학습 |
| S13 | [Reasoning Models Don't Always Say What They Think](https://arxiv.org/html/2505.05410v1) | 출력설명의인과한계 | 내부사고를노트로인증 |
| S14 | [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | shaping보장조건 제한 | 임의보상에정리적용 |
| S15 | [QLoRA](https://arxiv.org/html/2305.14314v1) | 4비트기저·저랭크학습 | RL성능/8GB보장 |
| S16 | [Transformers4.57.1](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes) | 고정환경양자화제약 | inference용auto배치를학습에무비판사용 |
| S17 | [PEFT0.17.1 source](https://raw.githubusercontent.com/huggingface/peft/v0.17.1/src/peft/utils/other.py) | 준비순서와requires_grad | 현재코드결함발견이라고단정 |
| S18 | [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) | native scene/program후보 | 무검사비중복선언 |
| S19 | [AI2D](https://arxiv.org/abs/1603.07396) | 도식관계annotation | 모든과학풀이검증가능주장 |
| S20 | [TabMWP/PromptPG](https://github.com/lupantech/PromptPG) | 표·답·원출처 | upstream풀이를SFTtarget사용 |

아직 검증하지 않은 것은 새 설계의 실제 생성 품질·원격 BF16 이식·독립 데이터 전수 중복 재검사·MMMU 점수 향상이다. 문헌 검토가 이 실행 증거를 대신하지 않는다.

<a id="D18"></a>
## D18. 현재 소스 함수·클래스 카탈로그

아래 부록은 실제 Python AST에서 추출한 현재 정의와 SHA다. `unannotated`는 반환형 annotation이 없다는 뜻이며 임의 타입을 지어내지 않았다. 구현 예정 repair_contract/repair_credit는 여기에 포함되지 않는다. private 함수 목록은 위치 탐색용이며 무조건 public API로 노출하라는 지시가 아니다. 함수의 실제 입력·출력 의미는 D06/D10 및 기존 코드의 검사와 함께 읽는다.

실제 파일 363개 중 생산·도구 Python 166개, 테스트 Python 197개를 구분했다. 아래 서명·반환 annotation·직접 호출·return 표현은 같은 SHA 소스의 AST에서 추출했다. 호출 목록은 정적 의존성 탐색용이며 실행 순서·분기 성공·완전한 의미를 인증하지 않는다. 긴 표현은 `…`로 표시하고 원문 파일을 연결했다. 새 작업자는 D10의 변경 대상만 수정하며 나머지 역사 실험 파일을 재설계하지 않는다.
### D18.001 `fine_tuning/__init__.py`

[현재 소스](../../fine_tuning/__init__.py) · SHA256 `11ded1cb3d26b8889ffa0e5ae65ccef66b570b65f43c594dd9c9ff2bf015bba9`

모듈의 기존 설명: Private fine-tuning overlay; never synchronized or pushed automatically.

최상위 함수·클래스 정의 없음. 초기화·상수·import는 원문을 따른다.

### D18.002 `fine_tuning/ablation.py`

[현재 소스](../../fine_tuning/ablation.py) · SHA256 `ea47f56ad04655033e6161035af2ea57cfdeb93a801610a17d05750577d4ad3f`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L33**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L53**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'rule_glyph'])).shuff …`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_canonical` — L70**
```python
render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: Repair only the Boolean query sequence; retain support/initial bit/options.
- 직접 호출: `exclusion_reason`, `ValueError`, `nvw.render`, `ImageDraw.Draw`, `draw.rectangle`, `range`, `len`, `enumerate`, `zip`, `nvw.arrow` 외 2개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L98**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L113**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 28개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L204**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L218**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L234**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.003 `fine_tuning/audit.py`

[현재 소스](../../fine_tuning/audit.py) · SHA256 `5fb3765042def4f2dce5c2d31da2ee739f25bb83063df2aee379bee55b9b8c13`

모듈의 기존 설명: Fail-closed provenance, exact/near duplicate and family-split training gate.

**`sha` — L20**
```python
sha(path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(path).open`, `Path`, `hashlib.file_digest(stream, 'sha256').hexdigest`, `hashlib.file_digest`
- 반환 표현: `hashlib.file_digest(stream, 'sha256').hexdigest()`

**`dump` — L25**
```python
dump(path, data) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `path.parent.mkdir`, `path.open`, `json.dump`, `stream.write`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`read_rows` — L33**
```python
read_rows(path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(path).open`, `Path`, `json.loads`, `line.strip`
- 반환 표현: `[json.loads(line) for line in stream if line.strip()]`

**`write_rows` — L38**
```python
write_rows(path, rows) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`normalized` — L46**
```python
normalized(text) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `re.sub('\\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip`, `re.sub`, `unicodedata.normalize('NFKC', text).casefold`, `unicodedata.normalize`
- 반환 표현: `re.sub('\\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()`

**`question_text` — L50**
```python
question_text(text) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `re.sub`, `unicodedata.normalize`, `re.split`, `normalized`
- 반환 표현: `normalized(value)`

**`text_hash` — L58**
```python
text_hash(text) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(question_text(text).encode()).hexdigest`, `hashlib.sha256`, `question_text(text).encode`, `question_text`
- 반환 표현: `hashlib.sha256(question_text(text).encode()).hexdigest()`

**`shingles` — L62**
```python
shingles(text, n=7) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `re.findall`, `question_text`, `' '.join`, `range`, `max`, `len`
- 반환 표현: `{' '.join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}`

**`image_fingerprint` — L67**
```python
image_fingerprint(image) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `image.convert`, `hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest`, `hashlib.sha256`, `str(rgb.size).encode`, `str`, `rgb.tobytes`, `rgb.convert('L').resize`, `rgb.convert`, `list`, `small.getdata` 외 4개
- 반환 표현: `{'pixels_sha256': pixel, 'dhash': f'{dhash:016x}', 'size': list(rgb.size)}`

**`check_lineage` — L77**
```python
check_lineage(row, sources) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `ValueError`, `isinstance`, `any`, `set`, `visit`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`check_splits` — L110**
```python
check_splits(rows) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `defaultdict`, `set`, `ValueError`, `ids.add`, `families[row['family_id']].add`, `roots[row['root_id']].add`, `any`, `len`, `list`, `families.values` 외 1개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`image_path` — L124**
```python
image_path(image_root, name) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `Path(name).is_absolute`, `Path`, `ValueError`, `Path(image_root).resolve`, `current.is_symlink`, `current.resolve`, `path.is_relative_to`, `path.is_file`
- 반환 표현: `path`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`model_input_record` — L139**
```python
model_input_record(row) -> unannotated
```
- 기존 설명: Remove only counted loader placeholders; retain the original source prompt.
- 직접 호출: `row.get`, `isinstance`, `text.strip`, `ValueError`, `original.count`, `len`, `text.count`, `original.replace('<image>', '').strip`, `original.replace`, `dict` 외 3개
- 반환 표현: `dict(row, user_text=stripped, original_user_text=original, original_user_text_sha256=hashlib.sha256(original.encode()).hexdigest())`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_fingerprint` — L157**
```python
input_fingerprint(row, image_root) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `model_input_record`, `sha`, `image_path`, `hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, ensure_ascii=False).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()`

**`training_input_contract` — L164**
```python
training_input_contract(protocol_path=None) -> unannotated
```
- 기존 설명: Only the frozen parent or its explicit canonical-bridge child is supported.
- 직접 호출: `Path(__file__).with_name`, `Path`, `sha`, `ValueError`, `json.loads`, `path.read_text`, `any`, `required.items`, `type`, `child.get` 외 1개
- 반환 표현: `contract`; `{**contract, **budgets, 'protocol_id': child['protocol_id'], 'protocol_sha256': sha(path), 'parent_protocol_sha256': parent_sha}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`processor_pixel_budgets` — L196**
```python
processor_pixel_budgets(image_processor) -> unannotated
```
- 기존 설명: Pinned Qwen processing gives the pair of pixel overrides priority over size.
- 직접 호출: `getattr`, `all`
- 반환 표현: `overrides`; `(image_processor.size['shortest_edge'], image_processor.size['longest_edge'])`

**`audit_model_inputs` — L204**
```python
audit_model_inputs(records_path, image_root, processor, output, sequence_budget=4096, episode_pixels=1048576, processor_revision=None, training_input_protocol=None) -> unannotated
```
- 기존 설명: Run the actual installed Qwen processor/collator; no model inference/download.
- 직접 호출: `isinstance`, `ValueError`, `Path`, `training_input_contract`, `processor_pixel_budgets`, `AssistantCollator`, `read_rows`, `model_input_record`, `image_path`, `collator` 외 13개
- 반환 표현: `receipt`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`preflight_records` — L259**
```python
preflight_records(records_path, image_root, model_path, output, sequence_budget=4096, episode_pixels=1048576, min_pixels=262144, max_pixels=1048576, training_input_protocol=None) -> unannotated
```
- 기존 설명: Pod-callable: local pinned processor only, sanitized records plus receipt.
- 직접 호출: `Path`, `model_path.is_dir`, `ValueError`, `training_input_contract`, `tuple`, `Qwen3VLProcessor.from_pretrained`, `str`, `audit_model_inputs`, `model_input_record`, `read_rows`
- 반환 표현: `([model_input_record(row) for row in read_rows(records_path)], receipt)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`check_target_identifiability` — L279**
```python
check_target_identifiability(row, image_root) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `ValueError`, `json.loads`, `isinstance`, `target.get`, `Path`, `canonical.is_file`, `read_rows`, `validate_episode`, `image_path` 외 8개
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`contamination_candidates` — L314**
```python
contamination_candidates(row, image_root, index) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `index.get`, `text_hash`, `reasons.append`, `shingles`, `matches.update`, `index.get('_inverted_shingles', {}).get`, `len`, `min`, `int` 외 7개
- 반환 표현: `sorted(set(reasons))`

**`approve` — L345**
```python
approve(records_path, image_root, sources, benchmark_index, output, input_preflight_receipt=None, training_input_protocol=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read_rows`, `check_splits`, `json.loads`, `Path(benchmark_index).read_text`, `Path`, `index.get`, `ValueError`, `index['coverage'].get`, `isinstance`, `str` 외 34개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.004 `fine_tuning/bridge_evaluation.py`

[현재 소스](../../fine_tuning/bridge_evaluation.py) · SHA256 `1c299d3d59d5c81353b1292dc78210d2e59b30fefd5ac02f86749a4684a1fdfe`

모듈의 기존 설명: Private common development probes for bounded synthetic Graph/CS and DC circuits.

**`graph_structure` — L55**
```python
graph_structure(value) -> unannotated
```
- 기존 설명: Gold-independent strict visible-node/topology schema; geometry is separate.
- 직접 호출: `isinstance`, `set`, `len`, `any`, `type`, `ValueError`, `math.isfinite`, `ids.add`, `links.add`, `dict` 외 1개
- 반환 표현: `dict(value, objects=sorted(value['objects'], key=lambda r: r['id']), relations=sorted(value['relations'], key=lambda r: (r['from'], r['to'])))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`target_structure` — L89**
```python
target_structure(scene) -> unannotated
```
- 기존 설명: Project observed render geometry, not solver states or final answers.
- 직접 호출: `graph_structure`
- 반환 표현: `graph_structure({'image_size': scene['image_size'], 'coordinate_system': scene['coordinate_system'], 'objects': [{key: item[key] for key in ('id', 'bbox', 'initially_active')} for item  …`

**`select_roots` — L96**
```python
select_roots(pairings, count=2, seed=3407) -> unannotated
```
- 기존 설명: Never read targets/outcomes or consume sealed roots to fill a quota.
- 직접 호출: `type`, `ValueError`, `audit.check_splits`, `dict`, `sorted`, `digest`, `len`
- 반환 표현: `[row['root_id'] for row in chosen]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare` — L109**
```python
prepare(pool, approvals, output, count=2, seed=3407) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `out.exists`, `out.is_symlink`, `FileExistsError`, `any`, `out.resolve().is_relative_to`, `out.resolve`, `source.resolve`, `ValueError`, `json.loads` 외 60개
- 반환 표현: `result`
- 명시적 예외: `FileExistsError`, `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L270**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `prepare`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.005 `fine_tuning/budget.py`

[현재 소스](../../fine_tuning/budget.py) · SHA256 `65dc0b1b1bf0aa1d7b6aee5fae3f38e0b2c06c1dcef2aa4f47d3f4480593c672`

모듈의 기존 설명: Standard-library owned-process watchdog, usable before dependency installation.

**`utc` — L14**
```python
utc(value: str) -> dt.datetime
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dt.datetime.fromisoformat`, `value.replace`, `ValueError`, `parsed.astimezone`
- 반환 표현: `parsed.astimezone(dt.timezone.utc)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`effective_deadline` — L21**
```python
effective_deadline(gpu_cutoff: dt.datetime, billing_start: dt.datetime, balance_usd: float, reserve_usd: float, hourly_usd: float) -> dt.datetime
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `all`, `math.isfinite`, `ValueError`, `dt.timedelta`, `min`
- 반환 표현: `min(gpu_cutoff, financial)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`terminate_group` — L31**
```python
terminate_group(process: subprocess.Popen, grace_seconds: float=15) -> None
```
- 기존 설명: Terminate a start_new_session child and kill residual workers after leader exit.
- 직접 호출: `math.isfinite`, `ValueError`, `os.killpg`, `process.wait`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L52**
```python
main() -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `time.monotonic`, `os.environ.get`, `ValueError`, `math.isfinite`, `effective_deadline`, `utc`, `min` 외 18개
- 반환 표현: `code`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

### D18.006 `fine_tuning/circuit_bridge.py`

[현재 소스](../../fine_tuning/circuit_bridge.py) · SHA256 `9ec218aec8ccc0e32b90441e88a0ea08647917caabfeaa5031e7b93c67072938`

모듈의 기존 설명: Quarantined independent DC circuit A0-A3 prototype, not full professional G7.

**`validate_world` — L32**
```python
validate_world(world) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `type`, `isinstance`, `len`, `any`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`solve_nodal` — L41**
```python
solve_nodal(world) -> unannotated
```
- 기존 설명: Exact KCL elimination; an ideal closed switch identifies nodes N2 and N3.
- 직접 호출: `validate_world`, `Fraction`, `zip`, `unknown.index`, `range`, `len`, `ValueError`, `dict`, `str`, `enumerate`
- 반환 표현: `{'source_current_A': str(currents[0]), 'node_voltages_V': {f'N{i}': str(nodes[i]) for i in range(4)}, 'resistor_currents_A': {f'C{i}': str(current) for i, current in enumerate(currents) …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_series_parallel` — L86**
```python
verify_series_parallel(world) -> unannotated
```
- 기존 설명: Separate circuit reduction, without constructing or solving nodal equations.
- 직접 호출: `validate_world`, `map`, `Fraction`, `str`
- 반환 표현: `{'source_current_A': str(total), 'node_voltages_V': {'N0': '0', 'N1': str(world['voltage_V']), 'N2': str(branch), 'N3': str(branch if world['switch_closed'] else Fraction(0))}, 'resisto …`

**`canonical_structure` — L100**
```python
canonical_structure(world) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_world`, `enumerate`, `zip`, `range`
- 반환 표현: `{'nodes': [f'N{i}' for i in range(4)], 'reference_node': 'N0', 'components': components}`

**`root_episode` — L112**
```python
root_episode(index, seed) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `nvw.digest`, `rng.choice`, `rng.sample`, `bool`, `rng.randrange`, `solve_nodal`, `verify_series_parallel`, `ValueError`, `math.gcd` 외 13개
- 반환 표현: `{'id': f'{VERSION}-{seed}-{index}', 'root_id': root_id, 'family_id': family, 'split': nvw.family_split(family), 'world': world, 'solution': solution, 'options': options, 'answer_letter' …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L144**
```python
render(episode, representation) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `set`, `sorted`, `mapping.values`, `canonical_structure`, `positions.items`, `Image.new`, `ImageDraw.Draw`, `label`, `draw.rectangle` 외 14개
- 반환 표현: `(image, {'representation': representation, 'image_size': [nvw.WIDTH, nvw.HEIGHT], 'canonical_structure': structure, 'node_positions': {n: list(p) for n, p in positions.items()}, 'symbol …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`record` — L230**
```python
record(episode, paths, view) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `len`, `json.dumps`, `canonical_structure`
- 반환 표현: `{'id': episode['id'] + '-' + episode['condition'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'condition': episode['condit …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`bundle` — L251**
```python
bundle(count, seed, start_index) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(v).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len` 외 33개
- 반환 표현: `files`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate` — L320**
```python
generate(output, count=4, seed=73041, start_index=1000000) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L333**
```python
verify(output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'root_count': manifest['root_count'], 'scope': SCOPE, **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L345**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.007 `fine_tuning/circuit_scoring.py`

[현재 소스](../../fine_tuning/circuit_scoring.py) · SHA256 `83d5a44dfa52aa4ac8e079d30af151787230680dfa94a000203f250865ab8da8`

모듈의 기존 설명: Strict CPU scoring for the bounded v2 Circuit response contract.

**`_object` — L25**
```python
_object(value: Any, keys: set[str]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `ValueError`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_unique_pairs` — L31**
```python
_unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_nonfinite` — L40**
```python
_nonfinite(value: str) -> Any
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_json` — L44**
```python
_json(raw: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `ValueError`, `type`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_letter` — L55**
```python
_letter(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_number` — L61**
```python
_number(value: Any) -> str
```
- 기존 설명: Normalize a finite exact scalar, with no unit or expression interpretation.
- 직접 호출: `type`, `ValueError`, `str`, `len`, `RATIO.fullmatch`, `Fraction`, `NUMBER.fullmatch`, `Decimal`, `decimal.as_tuple`, `decimal.is_finite` 외 2개
- 반환 표현: `str(Fraction(text))`; `str(Fraction(decimal))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_structure` — L83**
```python
_structure(value: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_object`, `type`, `len`, `any`, `set`, `ValueError`, `Counter`, `item.get`, `keys.add`, `seen.add` 외 2개
- 반환 표현: `dict(value, nodes=sorted(nodes), components=sorted(components, key=lambda item: item['id']))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_grounded` — L130**
```python
_grounded(value: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_object`, `_number`, `sorted`, `_letter`
- 반환 표현: `{'source_current_A': _number(value['source_current_A']), 'node_voltages_V': {node: _number(voltages[node]) for node in sorted(NODES)}, 'resistor_currents_A': {cid: _number(currents[cid] …`

**`extract_response` — L140**
```python
extract_response(raw: str, view: str, *, contract_version: str=VERSION) -> str | dict[str, Any]
```
- 기존 설명: Gold-independent extraction; unknown contracts and extra fields fail closed.
- 직접 호출: `ValueError`, `type`, `len`, `_letter`, `raw.strip`, `_json`, `_structure`, `_grounded`
- 반환 표현: `_letter(raw.strip())`; `_structure(value) if view == 'structure' else _grounded(value)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`score_circuit` — L157**
```python
score_circuit(raw: str, target: str, view: str, finish_reason: str='stop', *, contract_version: str=VERSION) -> dict[str, Any]
```
- 기존 설명: Compare with a validated target after extraction; malformed gold raises.
- 직접 호출: `ValueError`, `result.update`, `extract_response`, `isinstance`, `all`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.008 `fine_tuning/evaluate.py`

[현재 소스](../../fine_tuning/evaluate.py) · SHA256 `ddfb75bdff5dca72a27f99b048ad0961f77b1832311a62332e4a416b8eec4150`

모듈의 기존 설명: Private paired evaluation; immutable cohorts, fresh F0, unchanged public V8.

**`_parse_args` — L21**
```python
_parse_args() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_subparsers`, `sub.add_parser`, `prepare.add_argument`, `prepare_ext.add_argument`, `run.add_argument`, `parser.parse_args`
- 반환 표현: `parser.parse_args()`

**`digest` — L61**
```python
digest(value) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest`, `hashlib.sha256`, `json.dumps(value, ensure_ascii=False, sort_keys=True).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()`

**`sample_seed` — L65**
```python
sample_seed(sample_id, master=3407) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int.from_bytes`, `hashlib.sha256(f'{master}:{sample_id}'.encode()).digest`, `hashlib.sha256`, `f'{master}:{sample_id}'.encode`
- 반환 표현: `int.from_bytes(hashlib.sha256(f'{master}:{sample_id}'.encode()).digest()[:8], 'big') % 2 ** 31`

**`selected_ids` — L69**
```python
selected_ids(rows, per_subject=2, seed=3407) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `groups[row['subject']].append`, `any`, `len`, `set`, `groups.values`, `sorted`, `hashlib.sha256(f'{seed}:nvw-eval-v1:{sid}'.encode()).hexdigest`, `hashlib.sha256`, `f'{seed}:nvw-eval-v1:{sid}'.encode`
- 반환 표현: `[sample_id for subject in SUBJECTS for sample_id in sorted(groups[subject], key=lambda sid: hashlib.sha256(f'{seed}:nvw-eval-v1:{sid}'.encode()).hexdigest())[:per_subject]]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`inference_fields` — L81**
```python
inference_fields(row, subject) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ast.literal_eval`, `any`, `ValueError`, `row.get`, `row['hint'].strip`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare_mmmu` — L95**
```python
prepare_mmmu(data_root, output, per_subject=2) -> unannotated
```
- 기존 설명: Consume pinned validation only; select before consulting any gold/outcome.
- 직접 호출: `Path`, `json.loads`, `(root / 'manifest.json').read_text`, `ValueError`, `sha`, `path.stat`, `pq.read_table(path).to_pylist`, `pq.read_table`, `inference_fields`, `sample['id'].startswith` 외 27개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`visible_structure` — L163**
```python
visible_structure(value) -> unannotated
```
- 기존 설명: Drop renderer-internal glyph codes and duplicate line coordinates, not visible semantics.
- 직접 호출: `isinstance`, `visible_structure`, `value.items`
- 반환 표현: `{k: visible_structure(v) for k, v in value.items() if k not in {'glyph_code', 'segment'}}`; `[visible_structure(v) for v in value]`; `value`

**`prepare_external` — L172**
```python
prepare_external(nvw_root, trace_root, output, roots_per_family=4, trace_count=24) -> unannotated
```
- 기존 설명: Freeze development-only QA/structure/state tests before observing predictions.
- 직접 호출: `Path`, `read_rows`, `defaultdict`, `ValueError`, `groups.values`, `values['qa'][1]['family_id'].startswith`, `len`, `sorted`, `digest`, `set` 외 27개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`terminal_json` — L257**
```python
terminal_json(raw, *, parse_float=float) -> unannotated
```
- 기존 설명: No searching intermediate candidates or rescuing an answer with gold.
- 직접 호출: `str(raw).strip`, `str`, `text.startswith`, `re.sub`, `re.sub('\\n?\`\`\`$', '', text).strip`, `json.JSONDecoder`, `decoder.raw_decode`, `text[end:].strip`, `isinstance`, `ValueError`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`structure_metrics` — L280**
```python
structure_metrics(prediction, target) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Counter`, `facts`, `visible_structure`, `sum`, `(expected & observed).values`, `observed.values`, `expected.values`, `item.get`, `prediction.get`, `isinstance` 외 8개
- 반환 표현: `{'structure_precision': precision, 'structure_recall': recall, 'structure_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0, 'bbox_hits': hits, 'bbox_den …`

**`score_external` — L310**
```python
score_external(raw, gold, finish_reason) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `score_circuit`, `kind.removeprefix`, `extract_mcq`, `terminal_json`, `submitted.get`, `set`, `type`, `str(answer).strip`, `str`, `ValueError` 외 15개
- 반환 표현: `score_circuit(raw, gold['target'], kind.removeprefix('circuit_'), finish_reason, contract_version=gold['circuit_contract_version'])`; `{'correct': answer == gold['target'], 'extracted_answer': answer, 'parse_status': parsed['status'], 'metric_kind': kind}`; `{'correct': False, 'parse_status': 'NO_PARSE', 'metric_kind': kind}`; 외 3개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_cohort_images` — L386**
```python
verify_cohort_images(cohort, manifest) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(cohort).resolve`, `Path`, `read_rows`, `set`, `ValueError`, `path.resolve().is_relative_to`, `path.resolve`, `path.is_symlink`, `sha`, `len` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generation` — L413**
```python
generation(model, processor, record, image_root, protocol) -> unannotated
```
- 기존 설명: Strictly gold-blind model call; all outputs/costs are observed, not guessed.
- 직접 호출: `set`, `ValueError`, `(Path(image_root) / name).resolve`, `Path`, `Path(name).is_absolute`, `path.is_relative_to`, `Path(image_root).resolve`, `Image.open`, `images.append`, `image.convert` 외 25개
- 반환 표현: `{'id': record['id'], 'raw_response': processor.tokenizer.decode(output, skip_special_tokens=True), 'output_token_ids': output, 'finish_reason': finish, 'seed': seed, 'seconds': time.mon …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`resume_results` — L470**
```python
resume_results(parent, cohort, protocol_path, checkpoint, external) -> unannotated
```
- 기존 설명: Reuse only a closed budget-stopped prefix; never alter its original ledger.
- 직접 호출: `Path`, `json.loads`, `(parent / 'summary.json').read_text`, `sha`, `(cohort / 'manifest.json').read_text`, `summary.get`, `str`, `ValueError`, `read_rows`, `len` 외 1개
- 반환 표현: `(reused, summary)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`completed_histogram` — L503**
```python
completed_histogram(rows, field) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dict`, `Counter`
- 반환 표현: `dict(Counter((row[field] for row in rows if row['status'] == 'COMPLETED')))`

**`verify_transfer_checkpoint` — L507**
```python
verify_transfer_checkpoint(cohort_manifest, checkpoint) -> unannotated
```
- 기존 설명: A source-transfer cohort must exclude the evaluated checkpoint's training.
- 직접 호출: `cohort_manifest.get`, `json.loads`, `(Path(checkpoint) / 'metadata.json').read_text`, `Path`, `pairs.get`, `metadata.get`, `metadata.get('exposure', {}).get`, `ValueError`, `sha`
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`evaluate` — L525**
```python
evaluate(model_path, cohort, output, protocol_path, checkpoint=None, budget_seconds=14400, external=False, resume_from=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sha`, `Path`, `json.loads`, `Path(protocol_path).read_text`, `(cohort / 'manifest.json').read_text`, `ValueError`, `verify_cohort_images`, `verify_transfer_checkpoint`, `resume_results`, `out.exists` 외 34개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L622**
```python
main(args=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_parse_args`, `prepare_mmmu`, `prepare_external`, `print`, `json.dumps`, `evaluate`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.009 `fine_tuning/generated/boolean_ablation_prototype_v1/sources/ablation.py`

[현재 소스](../../fine_tuning/generated/boolean_ablation_prototype_v1/sources/ablation.py) · SHA256 `e1c681109f02f5b5ed8df6e6bcf05a0498a00a1f9373e7a60c2991eb0d9a64ae`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L31**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L51**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([VERSION, seed, episode['root_id'], 'rule_glyph'])).shuffle`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L68**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L83**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 27개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L172**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L186**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L202**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.010 `fine_tuning/generated/boolean_ablation_prototype_v1/sources/nvw.py`

[현재 소스](../../fine_tuning/generated/boolean_ablation_prototype_v1/sources/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.011 `fine_tuning/generated/boolean_ablation_prototype_v2/sources/ablation.py`

[현재 소스](../../fine_tuning/generated/boolean_ablation_prototype_v2/sources/ablation.py) · SHA256 `ea47f56ad04655033e6161035af2ea57cfdeb93a801610a17d05750577d4ad3f`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L33**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L53**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'rule_glyph'])).shuff …`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_canonical` — L70**
```python
render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: Repair only the Boolean query sequence; retain support/initial bit/options.
- 직접 호출: `exclusion_reason`, `ValueError`, `nvw.render`, `ImageDraw.Draw`, `draw.rectangle`, `range`, `len`, `enumerate`, `zip`, `nvw.arrow` 외 2개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L98**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L113**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 28개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L204**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L218**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L234**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.012 `fine_tuning/generated/boolean_ablation_prototype_v2/sources/nvw.py`

[현재 소스](../../fine_tuning/generated/boolean_ablation_prototype_v2/sources/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.013 `fine_tuning/generated/circuit_bridge_prototype_v1/sources/ablation.py`

[현재 소스](../../fine_tuning/generated/circuit_bridge_prototype_v1/sources/ablation.py) · SHA256 `ea47f56ad04655033e6161035af2ea57cfdeb93a801610a17d05750577d4ad3f`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L33**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L53**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'rule_glyph'])).shuff …`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_canonical` — L70**
```python
render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: Repair only the Boolean query sequence; retain support/initial bit/options.
- 직접 호출: `exclusion_reason`, `ValueError`, `nvw.render`, `ImageDraw.Draw`, `draw.rectangle`, `range`, `len`, `enumerate`, `zip`, `nvw.arrow` 외 2개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L98**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L113**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 28개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L204**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L218**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L234**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.014 `fine_tuning/generated/circuit_bridge_prototype_v1/sources/circuit_bridge.py`

[현재 소스](../../fine_tuning/generated/circuit_bridge_prototype_v1/sources/circuit_bridge.py) · SHA256 `377365db4a9643701fa6cb133dd7f40fe82857e7a91a78a67ddcbde2a44efc64`

모듈의 기존 설명: Quarantined independent DC circuit A0-A3 prototype, not full professional G7.

**`validate_world` — L30**
```python
validate_world(world) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `type`, `isinstance`, `len`, `any`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`solve_nodal` — L39**
```python
solve_nodal(world) -> unannotated
```
- 기존 설명: Exact KCL elimination; an ideal closed switch identifies nodes N2 and N3.
- 직접 호출: `validate_world`, `Fraction`, `zip`, `unknown.index`, `range`, `len`, `ValueError`, `dict`, `str`, `enumerate`
- 반환 표현: `{'source_current_A': str(currents[0]), 'node_voltages_V': {f'N{i}': str(nodes[i]) for i in range(4)}, 'resistor_currents_A': {f'C{i}': str(current) for i, current in enumerate(currents) …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_series_parallel` — L84**
```python
verify_series_parallel(world) -> unannotated
```
- 기존 설명: Separate circuit reduction, without constructing or solving nodal equations.
- 직접 호출: `validate_world`, `map`, `Fraction`, `str`
- 반환 표현: `{'source_current_A': str(total), 'node_voltages_V': {'N0': '0', 'N1': str(world['voltage_V']), 'N2': str(branch), 'N3': str(branch if world['switch_closed'] else Fraction(0))}, 'resisto …`

**`canonical_structure` — L98**
```python
canonical_structure(world) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_world`, `enumerate`, `zip`, `range`
- 반환 표현: `{'nodes': [f'N{i}' for i in range(4)], 'reference_node': 'N0', 'components': components}`

**`root_episode` — L110**
```python
root_episode(index, seed) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `nvw.digest`, `rng.choice`, `rng.sample`, `bool`, `rng.randrange`, `solve_nodal`, `verify_series_parallel`, `ValueError`, `math.gcd` 외 6개
- 반환 표현: `{'id': f'{VERSION}-{seed}-{index}', 'root_id': root_id, 'family_id': family, 'split': nvw.family_split(family), 'world': world, 'solution': solution, 'options': options, 'answer_letter' …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L131**
```python
render(episode, representation) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `set`, `sorted`, `mapping.values`, `canonical_structure`, `positions.items`, `Image.new`, `ImageDraw.Draw`, `label`, `draw.rectangle` 외 14개
- 반환 표현: `(image, {'representation': representation, 'image_size': [nvw.WIDTH, nvw.HEIGHT], 'canonical_structure': structure, 'node_positions': {n: list(p) for n, p in positions.items()}, 'symbol …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`record` — L217**
```python
record(episode, paths, view) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `len`, `json.dumps`, `canonical_structure`
- 반환 표현: `{'id': episode['id'] + '-' + episode['condition'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'condition': episode['condit …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`bundle` — L238**
```python
bundle(count, seed, start_index) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(v).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len` 외 33개
- 반환 표현: `files`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate` — L306**
```python
generate(output, count=4, seed=73041, start_index=1000000) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L319**
```python
verify(output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'root_count': manifest['root_count'], 'scope': SCOPE, **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L331**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.015 `fine_tuning/generated/circuit_bridge_prototype_v1/sources/graph_bridge.py`

[현재 소스](../../fine_tuning/generated/circuit_bridge_prototype_v1/sources/graph_bridge.py) · SHA256 `bea1438ed58ecf4a98c50494e543ceb0dce6003a9729898054088d30881eb43b`

모듈의 기존 설명: Quarantined independent Graph/CS representation bridge; partial G7 only.

**`font_sha256` — L34**
```python
font_sha256(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `hashlib.sha256(font.path.getvalue()).hexdigest`, `hashlib.sha256`, `font.path.getvalue`
- 반환 표현: `hashlib.sha256(font.path.getvalue()).hexdigest()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`shortcut_counts` — L40**
```python
shortcut_counts(world: dict[str, Any]) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `nvw.solve_signal`
- 반환 표현: `{'directed_final': len(nvw.solve_signal(world)[-1]), 'one_step_directed': len(nvw.solve_signal({**world, 'steps': 1})[-1]), 'one_step_undirected': len(nvw.solve_signal({**world, 'edges' …`

**`exclusion_reason` — L52**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `nvw.solve_signal`, `nvw.verify_signal`, `len`, `shortcut_counts`
- 반환 표현: `'UNSUPPORTED_NON_GRAPH_FAMILY'`; `'EMPTY_OR_UNINFORMATIVE_GRAPH'`; `'NO_POST_STEP_ONE_GROWTH'`; 외 2개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L77**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'endpoint_roles'])).s …`, `random.Random`, `nvw.digest`, `copy.deepcopy`, `range`, `label_rng.shuffle` 외 5개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`connector` — L107**
```python
connector(start: list[int], end: list[int]) -> tuple[list[float], ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `ValueError`, `tuple`
- 반환 표현: `tuple(([start[0] + ux * d, start[1] + uy * d] for d in (35, distance - 35, 75, distance - 75)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`segment_distance` — L117**
```python
segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `max`, `min`, `math.hypot`
- 반환 표현: `math.hypot(point[0] - start[0] - scale * dx, point[1] - start[1] - scale * dy)`

**`validate_layout` — L124**
```python
validate_layout(world: dict[str, Any], positions: list[list[int]]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `tuple`, `ValueError`, `any`, `connector`, `enumerate`, `zip`, `segment_distance`, `math.dist`, `markers.append`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`layout` — L152**
```python
layout(episode: dict[str, Any], variant: str) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `random.Random`, `nvw.digest`, `range`, `rng.randrange`, `rng.shuffle`, `validate_layout`
- 반환 표현: `positions`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L171**
```python
render(episode: dict[str, Any], representation: str, positions: list[list[int]]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `set`, `sorted`, `mapping.values`, `len`, `any`, `validate_layout`, `Image.new`, `ImageDraw.Draw` 외 15개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L251**
```python
paired_record(episode: dict[str, Any], paths: list[str], view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `json.loads`, `sorted`, `json.dumps`, `record.update`
- 반환 표현: `record`

**`bundle` — L270**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError` 외 32개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L373**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L387**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L402**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.016 `fine_tuning/generated/circuit_bridge_prototype_v1/sources/nvw.py`

[현재 소스](../../fine_tuning/generated/circuit_bridge_prototype_v1/sources/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.017 `fine_tuning/generated/circuit_bridge_prototype_v2/sources/ablation.py`

[현재 소스](../../fine_tuning/generated/circuit_bridge_prototype_v2/sources/ablation.py) · SHA256 `ea47f56ad04655033e6161035af2ea57cfdeb93a801610a17d05750577d4ad3f`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L33**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L53**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'rule_glyph'])).shuff …`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_canonical` — L70**
```python
render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: Repair only the Boolean query sequence; retain support/initial bit/options.
- 직접 호출: `exclusion_reason`, `ValueError`, `nvw.render`, `ImageDraw.Draw`, `draw.rectangle`, `range`, `len`, `enumerate`, `zip`, `nvw.arrow` 외 2개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L98**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L113**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 28개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L204**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L218**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L234**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.018 `fine_tuning/generated/circuit_bridge_prototype_v2/sources/circuit_bridge.py`

[현재 소스](../../fine_tuning/generated/circuit_bridge_prototype_v2/sources/circuit_bridge.py) · SHA256 `9ec218aec8ccc0e32b90441e88a0ea08647917caabfeaa5031e7b93c67072938`

모듈의 기존 설명: Quarantined independent DC circuit A0-A3 prototype, not full professional G7.

**`validate_world` — L32**
```python
validate_world(world) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `type`, `isinstance`, `len`, `any`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`solve_nodal` — L41**
```python
solve_nodal(world) -> unannotated
```
- 기존 설명: Exact KCL elimination; an ideal closed switch identifies nodes N2 and N3.
- 직접 호출: `validate_world`, `Fraction`, `zip`, `unknown.index`, `range`, `len`, `ValueError`, `dict`, `str`, `enumerate`
- 반환 표현: `{'source_current_A': str(currents[0]), 'node_voltages_V': {f'N{i}': str(nodes[i]) for i in range(4)}, 'resistor_currents_A': {f'C{i}': str(current) for i, current in enumerate(currents) …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_series_parallel` — L86**
```python
verify_series_parallel(world) -> unannotated
```
- 기존 설명: Separate circuit reduction, without constructing or solving nodal equations.
- 직접 호출: `validate_world`, `map`, `Fraction`, `str`
- 반환 표현: `{'source_current_A': str(total), 'node_voltages_V': {'N0': '0', 'N1': str(world['voltage_V']), 'N2': str(branch), 'N3': str(branch if world['switch_closed'] else Fraction(0))}, 'resisto …`

**`canonical_structure` — L100**
```python
canonical_structure(world) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_world`, `enumerate`, `zip`, `range`
- 반환 표현: `{'nodes': [f'N{i}' for i in range(4)], 'reference_node': 'N0', 'components': components}`

**`root_episode` — L112**
```python
root_episode(index, seed) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `nvw.digest`, `rng.choice`, `rng.sample`, `bool`, `rng.randrange`, `solve_nodal`, `verify_series_parallel`, `ValueError`, `math.gcd` 외 13개
- 반환 표현: `{'id': f'{VERSION}-{seed}-{index}', 'root_id': root_id, 'family_id': family, 'split': nvw.family_split(family), 'world': world, 'solution': solution, 'options': options, 'answer_letter' …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L144**
```python
render(episode, representation) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `set`, `sorted`, `mapping.values`, `canonical_structure`, `positions.items`, `Image.new`, `ImageDraw.Draw`, `label`, `draw.rectangle` 외 14개
- 반환 표현: `(image, {'representation': representation, 'image_size': [nvw.WIDTH, nvw.HEIGHT], 'canonical_structure': structure, 'node_positions': {n: list(p) for n, p in positions.items()}, 'symbol …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`record` — L230**
```python
record(episode, paths, view) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `len`, `json.dumps`, `canonical_structure`
- 반환 표현: `{'id': episode['id'] + '-' + episode['condition'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'condition': episode['condit …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`bundle` — L251**
```python
bundle(count, seed, start_index) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(v).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len` 외 33개
- 반환 표현: `files`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate` — L320**
```python
generate(output, count=4, seed=73041, start_index=1000000) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L333**
```python
verify(output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'root_count': manifest['root_count'], 'scope': SCOPE, **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L345**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.019 `fine_tuning/generated/circuit_bridge_prototype_v2/sources/graph_bridge.py`

[현재 소스](../../fine_tuning/generated/circuit_bridge_prototype_v2/sources/graph_bridge.py) · SHA256 `bea1438ed58ecf4a98c50494e543ceb0dce6003a9729898054088d30881eb43b`

모듈의 기존 설명: Quarantined independent Graph/CS representation bridge; partial G7 only.

**`font_sha256` — L34**
```python
font_sha256(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `hashlib.sha256(font.path.getvalue()).hexdigest`, `hashlib.sha256`, `font.path.getvalue`
- 반환 표현: `hashlib.sha256(font.path.getvalue()).hexdigest()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`shortcut_counts` — L40**
```python
shortcut_counts(world: dict[str, Any]) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `nvw.solve_signal`
- 반환 표현: `{'directed_final': len(nvw.solve_signal(world)[-1]), 'one_step_directed': len(nvw.solve_signal({**world, 'steps': 1})[-1]), 'one_step_undirected': len(nvw.solve_signal({**world, 'edges' …`

**`exclusion_reason` — L52**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `nvw.solve_signal`, `nvw.verify_signal`, `len`, `shortcut_counts`
- 반환 표현: `'UNSUPPORTED_NON_GRAPH_FAMILY'`; `'EMPTY_OR_UNINFORMATIVE_GRAPH'`; `'NO_POST_STEP_ONE_GROWTH'`; 외 2개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L77**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'endpoint_roles'])).s …`, `random.Random`, `nvw.digest`, `copy.deepcopy`, `range`, `label_rng.shuffle` 외 5개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`connector` — L107**
```python
connector(start: list[int], end: list[int]) -> tuple[list[float], ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `ValueError`, `tuple`
- 반환 표현: `tuple(([start[0] + ux * d, start[1] + uy * d] for d in (35, distance - 35, 75, distance - 75)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`segment_distance` — L117**
```python
segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `max`, `min`, `math.hypot`
- 반환 표현: `math.hypot(point[0] - start[0] - scale * dx, point[1] - start[1] - scale * dy)`

**`validate_layout` — L124**
```python
validate_layout(world: dict[str, Any], positions: list[list[int]]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `tuple`, `ValueError`, `any`, `connector`, `enumerate`, `zip`, `segment_distance`, `math.dist`, `markers.append`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`layout` — L152**
```python
layout(episode: dict[str, Any], variant: str) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `random.Random`, `nvw.digest`, `range`, `rng.randrange`, `rng.shuffle`, `validate_layout`
- 반환 표현: `positions`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L171**
```python
render(episode: dict[str, Any], representation: str, positions: list[list[int]]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `set`, `sorted`, `mapping.values`, `len`, `any`, `validate_layout`, `Image.new`, `ImageDraw.Draw` 외 15개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L251**
```python
paired_record(episode: dict[str, Any], paths: list[str], view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `json.loads`, `sorted`, `json.dumps`, `record.update`
- 반환 표현: `record`

**`bundle` — L270**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError` 외 32개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L373**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L387**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L402**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.020 `fine_tuning/generated/circuit_bridge_prototype_v2/sources/nvw.py`

[현재 소스](../../fine_tuning/generated/circuit_bridge_prototype_v2/sources/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.021 `fine_tuning/generated/graph_cs_bridge_prototype_v1/sources/ablation.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_prototype_v1/sources/ablation.py) · SHA256 `ea47f56ad04655033e6161035af2ea57cfdeb93a801610a17d05750577d4ad3f`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L33**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L53**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'rule_glyph'])).shuff …`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_canonical` — L70**
```python
render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: Repair only the Boolean query sequence; retain support/initial bit/options.
- 직접 호출: `exclusion_reason`, `ValueError`, `nvw.render`, `ImageDraw.Draw`, `draw.rectangle`, `range`, `len`, `enumerate`, `zip`, `nvw.arrow` 외 2개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L98**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L113**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 28개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L204**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L218**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L234**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.022 `fine_tuning/generated/graph_cs_bridge_prototype_v1/sources/graph_bridge.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_prototype_v1/sources/graph_bridge.py) · SHA256 `64873ce64a91e85c9c44e3d4b88a8836c14f8f817f3a3625803b15353a73eaab`

모듈의 기존 설명: Quarantined independent Graph/CS representation bridge; partial G7 only.

**`exclusion_reason` — L32**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `nvw.solve_signal`, `nvw.verify_signal`, `len`
- 반환 표현: `'UNSUPPORTED_NON_GRAPH_FAMILY'`; `'EMPTY_OR_UNINFORMATIVE_GRAPH'`; `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L52**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([VERSION, seed, episode['root_id'], 'endpoint_roles'])).shuffle`, `random.Random`, `nvw.digest`, `copy.deepcopy`, `range`, `label_rng.shuffle` 외 4개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`connector` — L81**
```python
connector(start: list[int], end: list[int]) -> tuple[list[float], ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `ValueError`, `tuple`
- 반환 표현: `tuple(([start[0] + ux * d, start[1] + uy * d] for d in (35, distance - 35, 75, distance - 75)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`segment_distance` — L91**
```python
segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `max`, `min`, `math.hypot`
- 반환 표현: `math.hypot(point[0] - start[0] - scale * dx, point[1] - start[1] - scale * dy)`

**`validate_layout` — L98**
```python
validate_layout(world: dict[str, Any], positions: list[list[int]]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `tuple`, `ValueError`, `any`, `connector`, `enumerate`, `zip`, `segment_distance`, `math.dist`, `markers.append`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`layout` — L126**
```python
layout(episode: dict[str, Any], variant: str) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `random.Random`, `nvw.digest`, `range`, `rng.randrange`, `rng.shuffle`, `validate_layout`
- 반환 표현: `positions`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L145**
```python
render(episode: dict[str, Any], representation: str, positions: list[list[int]]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `set`, `sorted`, `mapping.values`, `len`, `any`, `validate_layout`, `Image.new`, `ImageDraw.Draw` 외 14개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L224**
```python
paired_record(episode: dict[str, Any], paths: list[str], view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `json.loads`, `sorted`, `json.dumps`, `record.update`
- 반환 표현: `record`

**`bundle` — L243**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError` 외 35개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L343**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L357**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L372**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.023 `fine_tuning/generated/graph_cs_bridge_prototype_v1/sources/nvw.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_prototype_v1/sources/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.024 `fine_tuning/generated/graph_cs_bridge_prototype_v2/sources/ablation.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_prototype_v2/sources/ablation.py) · SHA256 `ea47f56ad04655033e6161035af2ea57cfdeb93a801610a17d05750577d4ad3f`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L33**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L53**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'rule_glyph'])).shuff …`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_canonical` — L70**
```python
render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: Repair only the Boolean query sequence; retain support/initial bit/options.
- 직접 호출: `exclusion_reason`, `ValueError`, `nvw.render`, `ImageDraw.Draw`, `draw.rectangle`, `range`, `len`, `enumerate`, `zip`, `nvw.arrow` 외 2개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L98**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L113**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 28개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L204**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L218**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L234**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.025 `fine_tuning/generated/graph_cs_bridge_prototype_v2/sources/graph_bridge.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_prototype_v2/sources/graph_bridge.py) · SHA256 `bea1438ed58ecf4a98c50494e543ceb0dce6003a9729898054088d30881eb43b`

모듈의 기존 설명: Quarantined independent Graph/CS representation bridge; partial G7 only.

**`font_sha256` — L34**
```python
font_sha256(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `hashlib.sha256(font.path.getvalue()).hexdigest`, `hashlib.sha256`, `font.path.getvalue`
- 반환 표현: `hashlib.sha256(font.path.getvalue()).hexdigest()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`shortcut_counts` — L40**
```python
shortcut_counts(world: dict[str, Any]) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `nvw.solve_signal`
- 반환 표현: `{'directed_final': len(nvw.solve_signal(world)[-1]), 'one_step_directed': len(nvw.solve_signal({**world, 'steps': 1})[-1]), 'one_step_undirected': len(nvw.solve_signal({**world, 'edges' …`

**`exclusion_reason` — L52**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `nvw.solve_signal`, `nvw.verify_signal`, `len`, `shortcut_counts`
- 반환 표현: `'UNSUPPORTED_NON_GRAPH_FAMILY'`; `'EMPTY_OR_UNINFORMATIVE_GRAPH'`; `'NO_POST_STEP_ONE_GROWTH'`; 외 2개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L77**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'endpoint_roles'])).s …`, `random.Random`, `nvw.digest`, `copy.deepcopy`, `range`, `label_rng.shuffle` 외 5개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`connector` — L107**
```python
connector(start: list[int], end: list[int]) -> tuple[list[float], ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `ValueError`, `tuple`
- 반환 표현: `tuple(([start[0] + ux * d, start[1] + uy * d] for d in (35, distance - 35, 75, distance - 75)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`segment_distance` — L117**
```python
segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `max`, `min`, `math.hypot`
- 반환 표현: `math.hypot(point[0] - start[0] - scale * dx, point[1] - start[1] - scale * dy)`

**`validate_layout` — L124**
```python
validate_layout(world: dict[str, Any], positions: list[list[int]]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `tuple`, `ValueError`, `any`, `connector`, `enumerate`, `zip`, `segment_distance`, `math.dist`, `markers.append`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`layout` — L152**
```python
layout(episode: dict[str, Any], variant: str) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `random.Random`, `nvw.digest`, `range`, `rng.randrange`, `rng.shuffle`, `validate_layout`
- 반환 표현: `positions`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L171**
```python
render(episode: dict[str, Any], representation: str, positions: list[list[int]]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `set`, `sorted`, `mapping.values`, `len`, `any`, `validate_layout`, `Image.new`, `ImageDraw.Draw` 외 15개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L251**
```python
paired_record(episode: dict[str, Any], paths: list[str], view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `json.loads`, `sorted`, `json.dumps`, `record.update`
- 반환 표현: `record`

**`bundle` — L270**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError` 외 32개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L373**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L387**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L402**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.026 `fine_tuning/generated/graph_cs_bridge_prototype_v2/sources/nvw.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_prototype_v2/sources/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.027 `fine_tuning/generated/graph_cs_bridge_split_pool32_v2/sources/ablation.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_split_pool32_v2/sources/ablation.py) · SHA256 `ea47f56ad04655033e6161035af2ea57cfdeb93a801610a17d05750577d4ad3f`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L33**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L53**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'rule_glyph'])).shuff …`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_canonical` — L70**
```python
render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: Repair only the Boolean query sequence; retain support/initial bit/options.
- 직접 호출: `exclusion_reason`, `ValueError`, `nvw.render`, `ImageDraw.Draw`, `draw.rectangle`, `range`, `len`, `enumerate`, `zip`, `nvw.arrow` 외 2개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L98**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L113**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 28개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L204**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L218**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L234**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.028 `fine_tuning/generated/graph_cs_bridge_split_pool32_v2/sources/graph_bridge.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_split_pool32_v2/sources/graph_bridge.py) · SHA256 `bea1438ed58ecf4a98c50494e543ceb0dce6003a9729898054088d30881eb43b`

모듈의 기존 설명: Quarantined independent Graph/CS representation bridge; partial G7 only.

**`font_sha256` — L34**
```python
font_sha256(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `hashlib.sha256(font.path.getvalue()).hexdigest`, `hashlib.sha256`, `font.path.getvalue`
- 반환 표현: `hashlib.sha256(font.path.getvalue()).hexdigest()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`shortcut_counts` — L40**
```python
shortcut_counts(world: dict[str, Any]) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `nvw.solve_signal`
- 반환 표현: `{'directed_final': len(nvw.solve_signal(world)[-1]), 'one_step_directed': len(nvw.solve_signal({**world, 'steps': 1})[-1]), 'one_step_undirected': len(nvw.solve_signal({**world, 'edges' …`

**`exclusion_reason` — L52**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `nvw.solve_signal`, `nvw.verify_signal`, `len`, `shortcut_counts`
- 반환 표현: `'UNSUPPORTED_NON_GRAPH_FAMILY'`; `'EMPTY_OR_UNINFORMATIVE_GRAPH'`; `'NO_POST_STEP_ONE_GROWTH'`; 외 2개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L77**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'endpoint_roles'])).s …`, `random.Random`, `nvw.digest`, `copy.deepcopy`, `range`, `label_rng.shuffle` 외 5개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`connector` — L107**
```python
connector(start: list[int], end: list[int]) -> tuple[list[float], ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `ValueError`, `tuple`
- 반환 표현: `tuple(([start[0] + ux * d, start[1] + uy * d] for d in (35, distance - 35, 75, distance - 75)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`segment_distance` — L117**
```python
segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `max`, `min`, `math.hypot`
- 반환 표현: `math.hypot(point[0] - start[0] - scale * dx, point[1] - start[1] - scale * dy)`

**`validate_layout` — L124**
```python
validate_layout(world: dict[str, Any], positions: list[list[int]]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `tuple`, `ValueError`, `any`, `connector`, `enumerate`, `zip`, `segment_distance`, `math.dist`, `markers.append`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`layout` — L152**
```python
layout(episode: dict[str, Any], variant: str) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `random.Random`, `nvw.digest`, `range`, `rng.randrange`, `rng.shuffle`, `validate_layout`
- 반환 표현: `positions`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L171**
```python
render(episode: dict[str, Any], representation: str, positions: list[list[int]]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `set`, `sorted`, `mapping.values`, `len`, `any`, `validate_layout`, `Image.new`, `ImageDraw.Draw` 외 15개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L251**
```python
paired_record(episode: dict[str, Any], paths: list[str], view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `json.loads`, `sorted`, `json.dumps`, `record.update`
- 반환 표현: `record`

**`bundle` — L270**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError` 외 32개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L373**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L387**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L402**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.029 `fine_tuning/generated/graph_cs_bridge_split_pool32_v2/sources/nvw.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_split_pool32_v2/sources/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.030 `fine_tuning/generated/graph_cs_bridge_split_pool_v2/sources/ablation.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_split_pool_v2/sources/ablation.py) · SHA256 `ea47f56ad04655033e6161035af2ea57cfdeb93a801610a17d05750577d4ad3f`

모듈의 기존 설명: Quarantined Boolean A0–A3 generation prototype; no external data/model inputs.

**`exclusion_reason` — L33**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: Reject ancestry/approval taint, then filter unsupported semantic roots.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `len`, `set`, `all`
- 반환 표현: `'UNSUPPORTED_NON_BOOLEAN_FAMILY'`; `'DUPLICATE_SEMANTIC_RULES'`; `'NOT_FULLY_IDENTIFIABLE'`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L53**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'rule_glyph'])).shuff …`, `random.Random`, `nvw.digest`, `dict`, `zip`, `copy.deepcopy` 외 1개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_canonical` — L70**
```python
render_canonical(episode: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: Repair only the Boolean query sequence; retain support/initial bit/options.
- 직접 호출: `exclusion_reason`, `ValueError`, `nvw.render`, `ImageDraw.Draw`, `draw.rectangle`, `range`, `len`, `enumerate`, `zip`, `nvw.arrow` 외 2개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L98**
```python
paired_record(episode: dict[str, Any], first: str, companion: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `record.update`
- 반환 표현: `record`

**`bundle` — L113**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: Pure deterministic artifact construction; intentionally no data-file argument.
- 직접 호출: `ValueError`, `Path`, `(source_dir / name).read_bytes`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError`, `nvw.build_episode` 외 28개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L204**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L218**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: Compare every byte to fresh deterministic generation, not just stored hashes.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L234**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.031 `fine_tuning/generated/graph_cs_bridge_split_pool_v2/sources/graph_bridge.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_split_pool_v2/sources/graph_bridge.py) · SHA256 `bea1438ed58ecf4a98c50494e543ceb0dce6003a9729898054088d30881eb43b`

모듈의 기존 설명: Quarantined independent Graph/CS representation bridge; partial G7 only.

**`font_sha256` — L34**
```python
font_sha256(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `hashlib.sha256(font.path.getvalue()).hexdigest`, `hashlib.sha256`, `font.path.getvalue`
- 반환 표현: `hashlib.sha256(font.path.getvalue()).hexdigest()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`shortcut_counts` — L40**
```python
shortcut_counts(world: dict[str, Any]) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `nvw.solve_signal`
- 반환 표현: `{'directed_final': len(nvw.solve_signal(world)[-1]), 'one_step_directed': len(nvw.solve_signal({**world, 'steps': 1})[-1]), 'one_step_undirected': len(nvw.solve_signal({**world, 'edges' …`

**`exclusion_reason` — L52**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `nvw.solve_signal`, `nvw.verify_signal`, `len`, `shortcut_counts`
- 반환 표현: `'UNSUPPORTED_NON_GRAPH_FAMILY'`; `'EMPTY_OR_UNINFORMATIVE_GRAPH'`; `'NO_POST_STEP_ONE_GROWTH'`; 외 2개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L77**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'endpoint_roles'])).s …`, `random.Random`, `nvw.digest`, `copy.deepcopy`, `range`, `label_rng.shuffle` 외 5개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`connector` — L107**
```python
connector(start: list[int], end: list[int]) -> tuple[list[float], ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `ValueError`, `tuple`
- 반환 표현: `tuple(([start[0] + ux * d, start[1] + uy * d] for d in (35, distance - 35, 75, distance - 75)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`segment_distance` — L117**
```python
segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `max`, `min`, `math.hypot`
- 반환 표현: `math.hypot(point[0] - start[0] - scale * dx, point[1] - start[1] - scale * dy)`

**`validate_layout` — L124**
```python
validate_layout(world: dict[str, Any], positions: list[list[int]]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `tuple`, `ValueError`, `any`, `connector`, `enumerate`, `zip`, `segment_distance`, `math.dist`, `markers.append`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`layout` — L152**
```python
layout(episode: dict[str, Any], variant: str) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `random.Random`, `nvw.digest`, `range`, `rng.randrange`, `rng.shuffle`, `validate_layout`
- 반환 표현: `positions`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L171**
```python
render(episode: dict[str, Any], representation: str, positions: list[list[int]]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `set`, `sorted`, `mapping.values`, `len`, `any`, `validate_layout`, `Image.new`, `ImageDraw.Draw` 외 15개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L251**
```python
paired_record(episode: dict[str, Any], paths: list[str], view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `json.loads`, `sorted`, `json.dumps`, `record.update`
- 반환 표현: `record`

**`bundle` — L270**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError` 외 32개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L373**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L387**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L402**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.032 `fine_tuning/generated/graph_cs_bridge_split_pool_v2/sources/nvw.py`

[현재 소스](../../fine_tuning/generated/graph_cs_bridge_split_pool_v2/sources/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.033 `fine_tuning/generated/pilot_pool_v1/generator_source.py`

[현재 소스](../../fine_tuning/generated/pilot_pool_v1/generator_source.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.034 `fine_tuning/generated/prototype_frozen/generator_source.py`

[현재 소스](../../fine_tuning/generated/prototype_frozen/generator_source.py) · SHA256 `1969cd2a2835cb62243d36fd4c6d7e017693552808eac07ad1f6969e25f73fef`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L347**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L398**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L408**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L441**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `verify_fsm`, `any`, `verify_signal` 외 3개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L475**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L581**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L640**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.035 `fine_tuning/generated/prototype_reviewed/generator_source.py`

[현재 소스](../../fine_tuning/generated/prototype_reviewed/generator_source.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.036 `fine_tuning/generated/replacement_pool_v1/generator_source.py`

[현재 소스](../../fine_tuning/generated/replacement_pool_v1/generator_source.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.037 `fine_tuning/graph_bridge.py`

[현재 소스](../../fine_tuning/graph_bridge.py) · SHA256 `bea1438ed58ecf4a98c50494e543ceb0dce6003a9729898054088d30881eb43b`

모듈의 기존 설명: Quarantined independent Graph/CS representation bridge; partial G7 only.

**`font_sha256` — L34**
```python
font_sha256(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `hashlib.sha256(font.path.getvalue()).hexdigest`, `hashlib.sha256`, `font.path.getvalue`
- 반환 표현: `hashlib.sha256(font.path.getvalue()).hexdigest()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`shortcut_counts` — L40**
```python
shortcut_counts(world: dict[str, Any]) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `nvw.solve_signal`
- 반환 표현: `{'directed_final': len(nvw.solve_signal(world)[-1]), 'one_step_directed': len(nvw.solve_signal({**world, 'steps': 1})[-1]), 'one_step_undirected': len(nvw.solve_signal({**world, 'edges' …`

**`exclusion_reason` — L52**
```python
exclusion_reason(episode: dict[str, Any]) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `nvw.validate_episode`, `nvw.solve_signal`, `nvw.verify_signal`, `len`, `shortcut_counts`
- 반환 표현: `'UNSUPPORTED_NON_GRAPH_FAMILY'`; `'EMPTY_OR_UNINFORMATIVE_GRAPH'`; `'NO_POST_STEP_ONE_GROWTH'`; 외 2개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`condition_episode` — L77**
```python
condition_episode(episode: dict[str, Any], arm: str, seed: int) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `list`, `FIXED_MAPPING.values`, `random.Random(nvw.digest([MAPPING_VERSION, seed, episode['root_id'], 'endpoint_roles'])).s …`, `random.Random`, `nvw.digest`, `copy.deepcopy`, `range`, `label_rng.shuffle` 외 5개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`connector` — L107**
```python
connector(start: list[int], end: list[int]) -> tuple[list[float], ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `ValueError`, `tuple`
- 반환 표현: `tuple(([start[0] + ux * d, start[1] + uy * d] for d in (35, distance - 35, 75, distance - 75)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`segment_distance` — L117**
```python
segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `max`, `min`, `math.hypot`
- 반환 표현: `math.hypot(point[0] - start[0] - scale * dx, point[1] - start[1] - scale * dy)`

**`validate_layout` — L124**
```python
validate_layout(world: dict[str, Any], positions: list[list[int]]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `tuple`, `ValueError`, `any`, `connector`, `enumerate`, `zip`, `segment_distance`, `math.dist`, `markers.append`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`layout` — L152**
```python
layout(episode: dict[str, Any], variant: str) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `random.Random`, `nvw.digest`, `range`, `rng.randrange`, `rng.shuffle`, `validate_layout`
- 반환 표현: `positions`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render` — L171**
```python
render(episode: dict[str, Any], representation: str, positions: list[list[int]]) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `exclusion_reason`, `ValueError`, `set`, `sorted`, `mapping.values`, `len`, `any`, `validate_layout`, `Image.new`, `ImageDraw.Draw` 외 15개
- 반환 표현: `(image, scene)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_record` — L251**
```python
paired_record(episode: dict[str, Any], paths: list[str], view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nvw.view_record`, `json.loads`, `sorted`, `json.dumps`, `record.update`
- 반환 표현: `record`

**`bundle` — L270**
```python
bundle(count: int, seed: int, start_index: int) -> dict[str, bytes]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `Path(__file__).with_name(name).read_bytes`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(data).hexdigest`, `hashlib.sha256`, `files.items`, `set`, `len`, `RuntimeError` 외 32개
- 반환 표현: `files`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`generate` — L373**
```python
generate(output: Path, count: int=4, seed: int=73041, start_index: int=1000000) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.exists`, `output.is_symlink`, `FileExistsError`, `bundle`, `output.mkdir`, `files.items`, `path.parent.mkdir`, `path.open`, `stream.write`, `json.loads`
- 반환 표현: `json.loads(files['manifest.json'])`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`verify` — L387**
```python
verify(output: Path) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.is_symlink`, `any`, `p.is_symlink`, `output.rglob`, `ValueError`, `json.loads`, `(output / 'manifest.json').read_bytes`, `bundle`, `p.relative_to(output).as_posix`, `p.relative_to` 외 4개
- 반환 표현: `{'status': 'CPU_REPRODUCED_NOT_APPROVED', 'scope': SCOPE, 'root_count': manifest['root_count'], **FLAGS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L402**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `verify`, `generate`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.038 `fine_tuning/modules.py`

[현재 소스](../../fine_tuning/modules.py) · SHA256 `ff642c875ba9ad1b9ab8064a01704c728982cf8399abe103743b12ea878229f0`

모듈의 기존 설명: Inspect and restore the actual Qwen3-VL selective training partitions.

**`inspect_modules` — L18**
```python
inspect_modules(model: nn.Module) -> dict
```
- 기존 설명: Derive roots from instantiated module types, never from advertised counts.
- 직접 호출: `dict`, `model.named_modules`, `named.items`, `type`, `len`, `ValueError`, `any`, `n.startswith`, `isinstance`, `name.rpartition` 외 10개
- 반환 표현: `inventory`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`parameter_group` — L53**
```python
parameter_group(name: str, inventory: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `name.removeprefix`, `any`, `name.startswith`
- 반환 표현: `'M'`; `'V' if name.startswith(inventory['visual'] + '.') else 'L'`

**`configure_arm` — L61**
```python
configure_arm(model: nn.Module, arm: str) -> tuple[nn.Module, dict, list[dict]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `hasattr`, `inspect_modules`, `model.requires_grad_`, `get_peft_model`, `cast`, `LoraConfig`, `model.named_modules`, `name.removeprefix`, `module.float().requires_grad_` 외 15개
- 반환 표현: `(model, inventory, groups)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`tensor_sha256` — L107**
```python
tensor_sha256(tensor: torch.Tensor) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hex …`, `hashlib.sha256`, `tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes`, `tensor.detach().cpu().contiguous().view(torch.uint8).numpy`, `tensor.detach().cpu().contiguous().view`, `tensor.detach().cpu().contiguous`, `tensor.detach().cpu`, `tensor.detach`
- 반환 표현: `hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()`

**`save_checkpoint` — L111**
```python
save_checkpoint(model: nn.Module, destination: Path, metadata: dict, inventory: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `destination.mkdir`, `hasattr`, `cast(Any, model).save_pretrained`, `cast`, `p.detach().cpu`, `p.detach`, `model.named_parameters`, `parameter_group`, `torch.save`, `tensor_sha256` 외 10개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`load_checkpoint` — L129**
```python
load_checkpoint(base_model: nn.Module, checkpoint: str | Path) -> nn.Module
```
- 기존 설명: Load complete adapter + all merger masters into a freshly loaded F0.
- 직접 호출: `Path`, `json.loads`, `(path / 'metadata.json').read_text`, `ValueError`, `metadata['files'].items`, `(path / name).resolve`, `target.is_relative_to`, `path.resolve`, `hashlib.sha256(target.read_bytes()).hexdigest`, `hashlib.sha256` 외 14개
- 반환 표현: `base_model`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.039 `fine_tuning/notebook_rl/__init__.py`

[현재 소스](../../fine_tuning/notebook_rl/__init__.py) · SHA256 `a7dbea7b146e91eebc3f0396597dc9b35a0aca4f4aff98052caa3295dd059474`

모듈의 기존 설명: Isolated, reward-only error-notebook RL; no training is run on import.

최상위 함수·클래스 정의 없음. 초기화·상수·import는 원문을 따른다.

### D18.040 `fine_tuning/notebook_rl/adapters/__init__.py`

[현재 소스](../../fine_tuning/notebook_rl/adapters/__init__.py) · SHA256 `4aedeec5a7966c476dc7a6f81ac6e3e2d3710ec52e80f3b72a1b70b39aa8b0e3`

모듈의 기존 설명: Source-to-RL record adapters; conversion is not approval for training.

최상위 함수·클래스 정의 없음. 초기화·상수·import는 원문을 따른다.

### D18.041 `fine_tuning/notebook_rl/adapters/geometry.py`

[현재 소스](../../fine_tuning/notebook_rl/adapters/geometry.py) · SHA256 `2772c4a996c412523b54208ff29212e078fd4c1af3f04340364a978c19bfa29c`

모듈의 기존 설명: Native Geometry3K/PGPS9K train records, never predicted parses or solution targets.

**`_hash` — L41**
```python
_hash(value: object) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, sepa …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ': …`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()`

**`_envelope` — L46**
```python
_envelope(record: dict, source: str, image_sha: str) -> tuple[dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `record.get`, `ValueError`, `provenance.get`, `isinstance`, `_SHA.fullmatch`, `re.fullmatch`
- 반환 표현: `(native, provenance)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_numeric_text` — L65**
```python
_numeric_text(value: object) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `value.strip`, `str`, `ValueError`, `number`
- 반환 표현: `text`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_precision` — L77**
```python
_precision(provenance: dict) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `provenance.get`, `type`, `set`, `ValueError`
- 반환 표현: `policy['round_to']`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_answer` — L87**
```python
_answer(native: dict, provenance: dict, *, g3k: bool) -> tuple[tuple[str, ...], AnswerType, str, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `native.get`, `type`, `len`, `ValueError`, `any`, `isinstance`, `v.strip`, `tuple`, `re.fullmatch`, `ord` 외 9개
- 반환 표현: `(choices, 'mcq', answer, {})`; `((), 'decimal', answer, {'round_to': precision})`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_slot` — L129**
```python
_slot(slots: dict[str, str], kind: str, labels: list[str], value: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `number`, `ValueError`, `len`, `sorted`, `min`, `list`, `reversed`, `''.join`, `str`
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_logic_slots` — L146**
```python
_logic_slots(logic: object) -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `any`, `isinstance`, `ValueError`, `re.sub`, `re.fullmatch`, `expression[expression.index('(', expression.index('(') + 1) + 1:-2].split`, `expression.index`, `expression.startswith`, `len` 외 4개
- 반환 표현: `dict(sorted(slots.items()))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_pgps_slots` — L172**
```python
_pgps_slots(clauses: object) -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `any`, `isinstance`, `ValueError`, `re.fullmatch`, `_slot`, `re.sub`, `re.findall`, `len`, `dict` 외 2개
- 반환 표현: `dict(sorted(slots.items()))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_records` — L192**
```python
_records(source: str, question: object, native: dict, provenance: dict, slots: dict[str, str], split: Split, image_rel: str, image_sha: str, mode: Mode, parents: tuple[str, ...]) -> tuple[ActorItem, VerifierItem, Lineage]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_answer`, `isinstance`, `question.strip`, `ValueError`, `ActorItem`, `VerifierItem`, `dict.fromkeys`, `tuple`, `slots.items`, `Lineage` 외 1개
- 반환 표현: `(actor, verifier, lineage)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`convert_geometry3k` — L215**
```python
convert_geometry3k(record: dict, split: Split, image_rel: str, image_sha: str, mode: Mode) -> tuple[ActorItem, VerifierItem, Lineage]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_envelope`, `native.get`, `type`, `str`, `ValueError`, `record.get`, `_logic_slots`, `diagram.get`, `_hash`, `_records`
- 반환 표현: `_records('geometry3k', native.get('problem_text'), native, provenance, slots, split, image_rel, image_sha, mode, (f'normalized-input-sha256:{proof}',))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`convert_pgps9k` — L232**
```python
convert_pgps9k(record: dict, split: Split, image_rel: str, image_sha: str, mode: Mode) -> tuple[ActorItem, VerifierItem, Lineage]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_envelope`, `ValueError`, `native.get`, `isinstance`, `PurePosixPath(diagram).is_absolute`, `PurePosixPath`, `any`, `diagram.split`, `provenance.get`, `type` 외 11개
- 반환 표현: `_records('pgps9k', native.get('text'), native, provenance, slots, split, image_rel, image_sha, mode, lineage_parents)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.042 `fine_tuning/notebook_rl/adapters/tabmwp_rl.py`

[현재 소스](../../fine_tuning/notebook_rl/adapters/tabmwp_rl.py) · SHA256 `0dcae024128801f38dbd5a12de92bfe4b3525ea8b7c4cc3fc7200dfa17fe4752`

모듈의 기존 설명: Native train annotations as verifier gold, never native solution targets.

**`convert` — L13**
```python
convert(native_problem: dict, pid: str, split: Split, image_rel: str, image_sha: str, mode: Mode) -> tuple[ActorItem, VerifierItem, Lineage]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `native_problem.get`, `ValueError`, `isinstance`, `pid.isdecimal`, `question.strip`, `answer.strip`, `len`, `any`, `c.strip`, `set` 외 20개
- 반환 표현: `(actor, verifier, lineage)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.043 `fine_tuning/notebook_rl/adjudication.py`

[현재 소스](../../fine_tuning/notebook_rl/adjudication.py) · SHA256 `ab69595b409e21a3a251c1a69652d5a600d2e410fe8186251dfe874255021bd9`

모듈의 기존 설명: Bind direct AI overlap reviews to complete scans; never repair missing scan gates.

**`candidate_id` — L23**
```python
candidate_id(candidate: dict) -> str
```
- 기존 설명: Stable content identity, including the exact similarity score/reference.
- 직접 호출: `type`, `ValueError`, `c._digest`
- 반환 표현: `c._digest(candidate)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_binding` — L30**
```python
_binding(files: dict[str, str] | None, path: Path, digest: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`, `path.absolute`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_artifact` — L38**
```python
_artifact(descriptor: dict, files: dict[str, str] | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `isinstance`, `Path(descriptor['path']).is_absolute`, `Path`, `_SHA.fullmatch`, `ValueError`, `_safe_root`, `_read`, `hashlib.sha256(raw).hexdigest` 외 3개
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_evidence` — L57**
```python
_evidence(descriptor: dict, files: dict[str, str] | None=None) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `isinstance`, `Path(descriptor['path']).is_absolute`, `Path`, `_SHA.fullmatch`, `ValueError`, `_safe_root`, `hashlib.sha256(_read(path)).hexdigest`, `hashlib.sha256` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_items_sha` — L69**
```python
_items_sha(items: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `items.items`, `isinstance`, `item.get`, `ValueError`, `list`, `any`, `_SHA.fullmatch`, `len`, `c._digest`
- 반환 표현: `c._digest(identity)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_complete` — L86**
```python
_complete(raw: dict, items: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_items_sha`, `raw.get`, `set`, `any`, `step.get`, `raw['steps'].values`, `type`, `len`, `c._digest`, `raw.items` 외 6개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_derive` — L111**
```python
_derive(raw: dict, review: dict, raw_descriptor: dict, review_descriptor: dict, items: dict, files: dict[str, str] | None=None) -> tuple[dict, list[dict]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_complete`, `review.get`, `any`, `type`, `ValueError`, `provenance.get`, `isinstance`, `provenance['reviewer_id'].strip`, `datetime.fromisoformat`, `provenance['reviewed_at_utc'].replace` 외 24개
- 반환 표현: `(result, effective)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`apply` — L165**
```python
apply(raw_receipt: dict, review: dict, items: dict) -> dict
```
- 기존 설명: Apply SHA-bound explicit decisions; originals are read, never modified.
- 직접 호출: `_artifact`, `_derive`
- 반환 표현: `result`

**`validate` — L172**
```python
validate(receipt: dict, items: dict, *, files: dict[str, str] | None=None) -> list[dict]
```
- 기존 설명: Reopen/rederive every binding; raw receipts retain all unreviewed warnings.
- 직접 호출: `type`, `receipt.get`, `ValueError`, `copy.deepcopy`, `_artifact`, `_derive`
- 반환 표현: `copy.deepcopy(receipt['candidates'])`; `copy.deepcopy(effective)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.044 `fine_tuning/notebook_rl/answers.py`

[현재 소스](../../fine_tuning/notebook_rl/answers.py) · SHA256 `426911e75ddbf1a278345cdea60058e8d97fc977b562ad7d3b1a3fcfacee3ab4`

모듈의 기존 설명: Strict RL answer contract; never searches reasoning for the reference answer.

**클래스 `Extraction` — L23**

기저: `object`.

```python
status: Literal['OK', 'INVALID_NO_FINAL', 'INVALID_MULTIPLE', 'TRUNCATED']
answer: str | None
```

**`_clean` — L28**
```python
_clean(value: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `value.strip`, `value.endswith`, `value[:-1].rstrip`, `range`, `value.startswith`, `len`, `value[2:-2].strip`, `value[1:-1].strip`, `re.fullmatch`, `value[7:-1].strip`
- 반환 표현: `value`

**`normalize_text` — L49**
```python
normalize_text(value: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `' '.join(unicodedata.normalize('NFKC', value).casefold().split()).removesuffix`, `' '.join`, `unicodedata.normalize('NFKC', value).casefold().split`, `unicodedata.normalize('NFKC', value).casefold`, `unicodedata.normalize`
- 반환 표현: `' '.join(unicodedata.normalize('NFKC', value).casefold().split()).removesuffix('.')`

**`extract_final` — L53**
```python
extract_final(text: str, finish_reason: str) -> Extraction
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `Extraction`, `text.rstrip`, `list`, `_FINAL.finditer`, `len`, `matches[0].end`, `sum`, `ch.isspace` 외 4개
- 반환 표현: `Extraction('TRUNCATED', None)`; `Extraction('INVALID_NO_FINAL', None)`; `Extraction('INVALID_MULTIPLE', None)`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`number` — L76**
```python
number(value: str) -> Fraction | None
```
- 기존 설명: Parse one exact decimal/rational; reject malformed commas and non-finite values.
- 직접 호출: `value.strip`, `len`, `re.fullmatch`, `value.replace`, `value.split`, `Fraction`, `int`, `left.strip`, `right.strip`, `Decimal` 외 4개
- 반환 표현: `None`; `Fraction(int(left.strip()), int(right.strip()))`; `Fraction(decimal)`

**`mcq_choice` — L95**
```python
mcq_choice(answer: str, choices: tuple[str, ...]) -> tuple[str | None, bool]
```
- 기존 설명: Return label and content-conflict flag without access to the reference label.
- 직접 호출: `re.fullmatch`, `answer.strip`, `ord`, `len`, `normalize_text`
- 반환 표현: `(None, False)`; `(label, conflict)`

**`is_ambiguous` — L108**
```python
is_ambiguous(answer: str, item: VerifierItem) -> bool
```
- 기존 설명: Type-aware delimiters: commas in 1,234 and slash in 3/4 are not alternatives.
- 직접 호출: `re.search`, `mcq_choice`, `number`, `re.fullmatch`, `answer.strip`
- 반환 표현: `True`; `False`; `conflict`

**`_numeric_equal` — L129**
```python
_numeric_equal(a: Fraction, g: Fraction, accept: dict) -> bool | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `type`, `Fraction`, `round`, `number`, `str`, `abs`
- 반환 표현: `a == g`; `None`; `round(a * scale) == round(g * scale)`; 외 1개 분기

**`_compare_typed` — L151**
```python
_compare_typed(answer: str, gold: str, kind: str, accept: dict) -> bool | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `number`, `re.fullmatch`, `bool`, `accept.get`, `type`, `answer.casefold`, `_numeric_equal`, `isinstance`, `any`, `v.strip` 외 15개
- 반환 표현: `None`; `bool(re.fullmatch(_INT, answer)) and a == g`; `False`; 외 7개 분기

**`compare` — L242**
```python
compare(answer: str, item: VerifierItem) -> bool | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `is_ambiguous`, `mcq_choice`, `_compare_typed`, `answer.strip`, `item.gold.strip`
- 반환 표현: `None`; `label == gold`; `_compare_typed(answer.strip(), item.gold.strip(), item.answer_type, item.accept)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.045 `fine_tuning/notebook_rl/approval.py`

[현재 소스](../../fine_tuning/notebook_rl/approval.py) · SHA256 `255c3814752d0e3984437f1564fc412bbdbdc324863c916d42d3c6f3dbb57cf4`

모듈의 기존 설명: Read-only launch validation, not an approval/report generator or GPU executor.

**클래스 `_Blocked` — L50**

기저: `Exception`.

**`_json` — L54**
```python
_json(value) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ': …`, `json.dumps`
- 반환 표현: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()`

**`_sha` — L58**
```python
_sha(raw: bytes) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`
- 반환 표현: `hashlib.sha256(raw).hexdigest()`

**`_hash` — L62**
```python
_hash(value) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256`, `set`, `update`, `_sha`, `_json`, `digest.hexdigest`
- 반환 표현: `_sha(_json(value))`; `digest.hexdigest()`

**`_zero` — L103**
```python
_zero(value) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `math.isfinite`
- 반환 표현: `type(value) in (int, float) and math.isfinite(value) and (value == 0)`

**`_require` — L107**
```python
_require(condition, code: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_Blocked`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `_Blocked`; 전파되는 callee 예외는 별도.

**`_read_json` — L112**
```python
_read_json(path: Path, files: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(path).absolute`, `Path`, `_safe_root`, `_read`, `json.loads`, `_require`, `type`, `str`, `_sha`
- 반환 표현: `value`

**`_artifact` — L124**
```python
_artifact(value: dict, files: dict) -> tuple[dict, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `_require`, `isinstance`, `_read_json`, `Path`, `str`, `Path(value['path']).absolute`, `_hash`
- 반환 표현: `(result, digest)`; `(value, _hash(value))`

**`code_hashes` — L135**
```python
code_hashes() -> dict[str, str]
```
- 기존 설명: Public production-code snapshot for the caller's live expected GPU scope.
- 직접 호출: `str`, `path.relative_to`, `_sha`, `_read`, `sorted`, `_HERE.rglob`
- 반환 표현: `{str(path.relative_to(_HERE)): _sha(_read(path)) for path in sorted(_HERE.rglob('*.py')) if 'data_tests' not in path.relative_to(_HERE).parts}`

**`_scanner_code` — L141**
```python
_scanner_code(code: dict, files: dict) -> None
```
- 기존 설명: Keep historical receipt paths, checking exact identities in this package.
- 직접 호출: `_require`, `type`, `len`, `set`, `code.items`, `re.fullmatch`, `Path`, `path.is_absolute`, `str`, `name.startswith` 외 8개
- 반환 표현: `None`

**`_time` — L175**
```python
_time(value) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `isinstance`, `datetime.fromisoformat`, `value.replace`, `stamp.utcoffset`, `timezone.utc.utcoffset`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_selection` — L181**
```python
_selection(selection: dict, phase: dict, phase_sha: str, build_sha: str, train: dict[str, ActorItem], structures: dict, protocol: dict) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `selection.get`, `phase.get`, `_time`, `type`, `all`, `isinstance`, `row.get`, `len`, `set` 외 15개
- 반환 표현: `sorted(selected)`

**`_proofs` — L248**
```python
_proofs(report: dict, prefix: str, count: int, binding: dict, scope: dict, producer: dict, files: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `report.get`, `type`, `set`, `range`, `gates.items`, `_artifact`, `proof.get`, `_hash`
- 반환 표현: `result`

**`_measurement_scope` — L269**
```python
_measurement_scope(scope: dict, binding: dict, protocol: dict, files: dict, *, limited_smoke: bool=False) -> None
```
- 기존 설명: Data-only scope, fixed BEFORE Phase A; never a GPU/training certificate.
- 직접 호출: `keys.add`, `_require`, `type`, `set`, `isinstance`, `bool`, `authorization['approval_reference'].strip`, `_time`, `code_hashes`, `Path(producer['path']).absolute` 외 5개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_human` — L300**
```python
_human(proof: dict, registry: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `set`, `registry.get`, `reviewers.get`, `reviewer.get`, `_time`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_ai_program_review` — L312**
```python
_ai_program_review(proof: dict, registry: dict) -> None
```
- 기존 설명: Declared AI review, never a human-review certificate or label producer.
- 직접 호출: `proof.get`, `type`, `_Blocked`, `_require`, `set`, `registry.get`, `reviewers.get`, `reviewer.get`, `_time`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `_Blocked`; 전파되는 callee 예외는 별도.

**`_data_gates` — L331**
```python
_data_gates(proofs: dict, items: dict, manifests: dict, input_files: dict, registry: dict, receipt_sha: str, review_seed: int, *, limited_smoke: bool=False) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `items.items`, `_require`, `len`, `m.get`, `_hash`, `type`, `proofs['D2']['measurements'].get`, `_ai_program_review if limited_smoke else _human`, `proofs['D3']['measurements'].get` 외 20개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_gpu_scope` — L388**
```python
_gpu_scope(scope: dict, binding: dict, protocol: dict, files: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `set`, `code_hashes`, `isinstance`, `resource_path`, `_read_json`, `str`, `_sha`, `_read` 외 7개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_gpu_gates` — L426**
```python
_gpu_gates(proofs: dict, scope: dict, protocol: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `proofs.items`, `_require`, `all`, `m['G0'].get`, `dict`, `m['G1'].get`, `type`, `m['G1']['trainable_counts'].values`, `m['G2'].get`, `_zero` 외 7개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_validate` — L475**
```python
_validate(payload: dict) -> dict
```
- 기존 설명: Trusted worker implementation; deliberately not called by the public actor facade.
- 직접 호출: `_safe_root`, `Path`, `payload.get`, `_require`, `_read_json`, `str`, `load_protocol`, `protocol.get`, `code_hashes`, `manifest.get` 외 64개
- 반환 표현: `result`

**`_public_validate` — L661**
```python
_public_validate(payload: dict, timeout: float) -> dict
```
- 기존 설명: One trusted worker for either stage; never read private files in the actor.
- 직접 호출: `payload.get`, `type`, `math.isfinite`, `_json`, `_require`, `len`, `subprocess.run`, `json.loads`, `result.get`, `set` 외 3개
- 반환 표현: `blocked | {'blockers': ['INVALID_TIMEOUT']}`; `result`; `blocked | {'blockers': [str(exc)]}`; 외 2개 분기

**`validate_for_training` — L703**
```python
validate_for_training(build_root: Path, selection_path: Path, *, phase_a_report: Path, index: dict, source_registry: dict, evidence: dict, data_report: dict | None, gpu_report: dict | None, expected_gpu_scope: dict | None, timeout: float=120) -> dict
```
- 기존 설명: No writes, approval creation, private actor-side reads or in-process fallback.
- 직접 호출: `_public_validate`, `str`, `Path(build_root).absolute`, `Path`, `Path(selection_path).absolute`, `Path(phase_a_report).absolute`
- 반환 표현: `_public_validate({'build_root': str(Path(build_root).absolute()), 'selection_path': str(Path(selection_path).absolute()), 'phase_a_report': str(Path(phase_a_report).absolute()), 'index' …`

**`validate_for_measurement` — L714**
```python
validate_for_measurement(build_root: Path, *, index: dict, source_registry: dict, evidence: dict, data_report: dict | None, expected_measurement_scope: dict | None, timeout: float=120) -> dict
```
- 기존 설명: Validate D1–D8 for the complete candidate pool before Phase A exists.
- 직접 호출: `_public_validate`, `str`, `Path(build_root).absolute`, `Path`
- 반환 표현: `_public_validate({'stage': 'measurement', 'build_root': str(Path(build_root).absolute()), 'index': index, 'source_registry': source_registry, 'evidence': evidence, 'data_report': data_r …`

**`validate_for_limited_smoke` — L729**
```python
validate_for_limited_smoke(build_root: Path, *, index: dict, source_registry: dict, evidence: dict, data_report: dict | None, expected_measurement_scope: dict | None, timeout: float=120) -> dict
```
- 기존 설명: Explicit opt-in: <=64 program-only records, every record AI reviewed.
- 직접 호출: `_public_validate`, `str`, `Path(build_root).absolute`, `Path`
- 반환 표현: `_public_validate({'stage': 'limited_smoke', 'build_root': str(Path(build_root).absolute()), 'index': index, 'source_registry': source_registry, 'evidence': evidence, 'data_report': data …`

**`validate_for_gate_data` — L744**
```python
validate_for_gate_data(build_root: Path, **request) -> dict
```
- 기존 설명: Only gate_session/G13 dispatch here; full measurement never opts in.
- 직접 호출: `request.get`, `type`, `scope.get`, `request.setdefault`, `validate_for_limited_smoke if limited else validate_for_measurement`
- 반환 표현: `(validate_for_limited_smoke if limited else validate_for_measurement)(build_root, **request)`

### D18.046 `fine_tuning/notebook_rl/attribution.py`

[현재 소스](../../fine_tuning/notebook_rl/attribution.py) · SHA256 `17bfce21f71344dc9c1cbd674631dd88a69dc47a600985be157a19713c2434fa`

모듈의 기존 설명: Functional evidence attribution, never a diagnosis of an internal module.

**`attempt_cause` — L11**
```python
attempt_cause(result: VerifierResult, item: VerifierItem) -> Cause | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `_check_slots`, `len`, `slot_id`, `item.slot_types.items`, `any`, `required_observations_verified`
- 반환 표현: `None`; `'FINALIZATION'`; `'UNKNOWN'`; 외 3개 분기

**`_check_slots` — L34**
```python
_check_slots(result: VerifierResult, item: VerifierItem) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `slot_id`, `len`, `set`, `ValueError`, `native_absent_relations`
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_probe_cause` — L49**
```python
_probe_cause(probe: dict, item: VerifierItem) -> str
```
- 기존 설명: Counts from separate diagnostic calls; raw gold facts are not accepted here.
- 직접 호출: `isinstance`, `set`, `ValueError`, `probe.get`, `type`, `evidence.append`, `len`, `known.pop`
- 반환 표현: `known.pop() if len(known) == 1 else 'UNKNOWN'`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`problem_route` — L77**
```python
problem_route(results: list[VerifierResult], items: VerifierItem, rule: str, probe: dict | None=None) -> tuple[Route, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `ValueError`, `enumerate`, `_check_slots`, `false_relations.append`, `list`, `len`, `evidence.append`, `required_observations_verified`, `comparable.append` 외 1개
- 반환 표현: `(route, {'rule': rule, 'observed_evidence': observed, 'wrong_required': evidence, 'false_relations': false_relations, 'complete_required_indices': comparable, 'probe_evidence': extra, ' …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`diagnosis_target` — L108**
```python
diagnosis_target(first_attempt: VerifierResult, item: VerifierItem) -> DiagTarget | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `attempt_cause`, `DiagTarget`, `verified_wrong_slots`
- 반환 표현: `None`; `DiagTarget('VISUAL', verified_wrong_slots(first_attempt.slots))`; `DiagTarget('FINALIZATION' if cause == 'FINALIZATION' else 'NON_VISUAL', ())`

**`random_route` — L118**
```python
random_route(rate: float, key: str) -> Route
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `math.isfinite`, `isinstance`, `ValueError`, `int.from_bytes`, `hashlib.sha256(key.encode()).digest`, `hashlib.sha256`, `key.encode`
- 반환 표현: `'VISION_ON' if value < rate else 'VISION_OFF'`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.047 `fine_tuning/notebook_rl/batches.py`

[현재 소스](../../fine_tuning/notebook_rl/batches.py) · SHA256 `37c0c11a57290626662c76b2906e73210bbbcb1e302d020ad59b027edc52279f`

모듈의 기존 설명: Bounded family-balanced rollout collection; this is not a data approval gate.

**`family_order` — L18**
```python
family_order(actors: list[ActorItem], seed: int) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `len`, `ValueError`, `defaultdict`, `to_json`, `families[actor.family].append`, `deque`, `sorted`, `families.items`, `any` 외 3개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `CollectedBatch` — L37**

기저: `object`.

```python
trees: list[ProblemTree]
images: dict[str, list]
cursor: int
attempted: list[str]
discarded: list[dict]
diagnosis: dict | None
seconds: float
status: str
```

**`collect_batch` — L48**
```python
collect_batch(actors: dict[str, ActorItem], structure_hashes: dict[str, str], order: list[str], cursor: int, loader: Callable[[ActorItem], list], engine: Any, verifier: Any, prompts: dict, protocol: dict, algorithm: str, *, should_stop: Callable[[], bool], seed: int=3407, on_tree: Callable[[ProblemTree], None] | None=None, on_group: Callable[[Group], None] | None=None) -> CollectedBatch
```
- 기존 설명: Consume at most 3B IDs once; record every skipped ID rather than retry forever.
- 직접 호출: `set`, `len`, `any`, `re.fullmatch`, `structure_hashes.values`, `type`, `ValueError`, `RolloutConfig.from_protocol`, `time.monotonic`, `should_stop` 외 18개
- 반환 표현: `CollectedBatch(trees, images, cursor, attempted, discarded, diagnosis, time.monotonic() - started, status)`
- 명시적 예외: `ValueError`, `re-raise`, `RuntimeError`; 전파되는 callee 예외는 별도.

### D18.048 `fine_tuning/notebook_rl/build_exclusion_index.py`

[현재 소스](../../fine_tuning/notebook_rl/build_exclusion_index.py) · SHA256 `72a5bf2962f8c6f0076bfd6ae075b09b41d26092382632fc48318ec2b30653fe`

모듈의 기존 설명: Build the private v2 native benchmark index on CPU; default is a read-only plan.

**`_inventory` — L30**
```python
_inventory() -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `vault._read`, `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`, `ValueError`, `json.loads`
- 반환 표현: `json.loads(raw, object_pairs_hook=vault._unique)['files']`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_private_root` — L37**
```python
_private_root(path: Path, *, create: bool=False) -> Path
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(path).absolute`, `Path`, `path.resolve().is_relative_to`, `path.resolve`, `ValueError`, `vault._safe_root`
- 반환 표현: `vault._safe_root(path, create=create)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_cache_path` — L44**
```python
_cache_path(entry: dict, cache_root: Path) -> Path | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `entry['dataset_id'].replace`, `candidate.exists`, `candidate.resolve`, `path.is_relative_to`, `cache_root.resolve`, `path.is_file`, `ValueError`, `candidate.is_symlink`
- 반환 표현: `path`; `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_verify_file` — L58**
```python
_verify_file(path: Path, entry: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.stat`, `audit.sha`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`plan` — L63**
```python
plan(cache_root: Path, output_root: Path) -> dict
```
- 기존 설명: Report cached presence only; execute always verifies bytes and row counts.
- 직접 호출: `output_root.resolve().is_relative_to`, `output_root.resolve`, `cache_root.resolve().is_relative_to`, `cache_root.resolve`, `ValueError`, `_cache_path`, `_inventory`, `sum`, `str`, `output_root.absolute`
- 반환 표현: `{'status': 'PLAN_ONLY', 'files': files, 'record_count': sum((e['records'] for e in files)), 'native_bytes': sum((e['bytes'] for e in files)), 'missing_download_bytes': sum((e['bytes'] f …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_write_once` — L75**
```python
_write_once(path: Path, raw: bytes) -> None
```
- 기존 설명: Publish atomically without replacing a prior payload, including on restart.
- 직접 호출: `path.exists`, `path.is_symlink`, `vault._read`, `ValueError`, `tempfile.mkstemp`, `os.fdopen`, `stream.write`, `stream.flush`, `os.fsync`, `stream.fileno` 외 2개
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_texts` — L96**
```python
_texts(row: dict, vision: bool) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `isinstance`, `question.strip`, `ValueError`, `ast.literal_eval`, `any`, `list`
- 반환 표현: `{'question': [question], 'choices': list(options), 'explanation': [] if explanation is None else [explanation]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_build_shard` — L119**
```python
_build_shard(entry: dict, path: Path, output: Path, images: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `pq.ParquetFile`, `ValueError`, `required.issubset`, `len`, `set`, `_IMAGE.fullmatch`, `native.iter_batches`, `batch.to_pylist`, `isinstance`, `ids.add` 외 15개
- 반환 표현: `{'records': records, 'images': owned_images}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_build` — L174**
```python
_build(entries: list[dict], cache_root: Path, output: Path, *, max_download_bytes: int=0, max_files: int | None=None, partial_name: str | None=None) -> dict
```
- 기존 설명: Internal partial entrypoint is for explicitly named synthetic fixture checks.
- 직접 호출: `type`, `ValueError`, `cache_root.absolute`, `cache_root.resolve().is_relative_to`, `cache_root.resolve`, `_private_root`, `vault._safe_root`, `audit.sha`, `shard_path.exists`, `shard_path.is_symlink` 외 22개
- 반환 표현: `{'status': 'CHECKPOINTED', 'completed_files': len(coverage), 'record_count': len(records), 'downloaded_bytes': download_bytes, 'training_allowed': False}`; `{'status': 'INDEX_READY' if prepared['full_coverage'] else 'NAMED_PARTIAL_INDEX_READY', 'index': str(output / 'index.json'), 'record_count': count, 'image_count': value['image_count'],  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L252**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `_build`, `_inventory`, `plan`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.049 `fine_tuning/notebook_rl/checkpoint.py`

[현재 소스](../../fine_tuning/notebook_rl/checkpoint.py) · SHA256 `7f82575b1bf57967a21a8a75531c2ed7ff480f9936c119a7f9b5b328583c97f0`

모듈의 기존 설명: Write-once, fully bound RL checkpoints; restoration always starts from fresh F0.

**`digest` — L26**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `json.dumps`, `hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':') …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=Fal …`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()`

**`file_hash` — L32**
```python
file_hash(path: Path) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256`, `path.open`, `iter`, `stream.read`, `value.update`, `value.hexdigest`
- 반환 표현: `value.hexdigest()`

**`_binding` — L40**
```python
_binding(protocol: dict, binding: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `any`, `isinstance`, `re.fullmatch`, `binding.values`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`capture_rng` — L48**
```python
capture_rng() -> dict
```
- 기존 설명: Safe-loadable representation; CUDA is never initialized merely to save CPU state.
- 직접 호출: `np.random.get_state`, `random.getstate`, `torch.tensor`, `keys.astype`, `torch.get_rng_state`, `torch.cuda.is_initialized`, `torch.cuda.get_rng_state_all`
- 반환 표현: `{'python': random.getstate(), 'numpy': {'algorithm': algorithm, 'keys': torch.tensor(keys.astype(np.int64)), 'position': position, 'has_gauss': has_gauss, 'cached_gaussian': cached}, 't …`

**`restore_rng` — L59**
```python
restore_rng(rng: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `ValueError`, `bool`, `((numpy['keys'] >= 0) & (numpy['keys'] < 2 ** 32)).all`, `math.isfinite`, `isinstance`, `any`, `torch.cuda.is_available`, `len`, `torch.cuda.device_count` 외 13개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_optimizer` — L90**
```python
make_optimizer(bundle: PolicyBundle, protocol: dict) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]
```
- 기존 설명: Stable L/M/V order, including route-OFF parameters, with no implicit LR scaling.
- 직접 호출: `ValueError`, `type`, `torch.optim.AdamW`, `tuple`, `torch.optim.lr_scheduler.LambdaLR`, `partial`
- 반환 표현: `(optimizer, scheduler)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_warmup_multiplier` — L106**
```python
_warmup_multiplier(step: int, *, warmup: int) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`
- 반환 표현: `min(1.0, (step + 1) / warmup) if warmup else 1.0`

**`_parameters` — L110**
```python
_parameters(bundle: PolicyBundle) -> dict[str, torch.nn.Parameter]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dict`, `bundle.model.named_parameters`, `len`, `set`, `ValueError`, `id`, `expected.values`, `bundle.params.values`, `list`, `str` 외 3개
- 반환 표현: `expected`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_optimizer_names` — L129**
```python
_optimizer_names(bundle: PolicyBundle, optimizer: torch.optim.Optimizer) -> list[list[str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_parameters`, `id`, `parameters.items`, `ValueError`, `len`, `set`
- 반환 표현: `groups`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_cursor` — L142**
```python
_cursor(cursor: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `cursor.keys`, `any`, `type`, `isinstance`, `ValueError`, `digest`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_optimizer` — L151**
```python
_validate_optimizer(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR, protocol: dict, step: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `make_optimizer`, `type`, `_optimizer_names`, `ValueError`, `zip`, `actual.items`, `expected.items`, `scheduler.state_dict`, `expected_sched.state_dict`, `any` 외 14개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`save` — L186**
```python
save(bundle: PolicyBundle, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR, step: int, cursor: dict, rng: dict, out: Path, *, protocol: dict, binding: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_binding`, `_cursor`, `type`, `ValueError`, `assert_base_unchanged`, `_parameters`, `any`, `parameters.values`, `_validate_optimizer`, `capture_rng` 외 22개
- 반환 표현: `metadata`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`read_metadata` — L225**
```python
read_metadata(ckpt: Path, *, binding: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `ckpt.is_dir`, `ckpt.is_symlink`, `ckpt.iterdir`, `any`, `p.is_file`, `p.is_symlink`, `ValueError`, `(ckpt / 'metadata.sha256').read_text(encoding='ascii').strip`, `(ckpt / 'metadata.sha256').read_text` 외 9개
- 반환 표현: `metadata`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_load` — L249**
```python
_load(model_path: Path, ckpt: Path, binding: dict, factory: PolicyFactory | None) -> tuple[PolicyBundle, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read_metadata`, `factory or build_policy`, `Path`, `ValueError`, `_parameters`, `load_file`, `str`, `set`, `rows.items`, `(adapters | mergers).items` 외 8개
- 반환 표현: `(bundle, metadata)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_eval` — L278**
```python
load_eval(model_path: Path, ckpt: Path, *, binding: dict, policy_factory: PolicyFactory | None=None, _test_policy_factory: PolicyFactory | None=None) -> PolicyBundle
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `_load`, `Path`, `bundle.model.eval`
- 반환 표현: `bundle`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_resume` — L288**
```python
load_resume(model_path: Path, ckpt: Path, protocol: dict, *, binding: dict, device: str | torch.device='cpu', policy_factory: PolicyFactory | None=None, _test_policy_factory: PolicyFactory | None=None) -> tuple[PolicyBundle, torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR, int, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_binding`, `read_metadata`, `json.loads`, `json.dumps`, `ValueError`, `_load`, `Path`, `bundle.model.to`, `torch.device`, `make_optimizer` 외 8개
- 반환 표현: `(bundle, optimizer, scheduler, metadata['step'], metadata['cursor'])`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.050 `fine_tuning/notebook_rl/circuit_trace.py`

[현재 소스](../../fine_tuning/notebook_rl/circuit_trace.py) · SHA256 `a87cfbf8b92f4c09d96a02c350fee31d8cdf118faa13fcee2f2fef5bc3162382`

모듈의 기존 설명: Bounded, gold-independent checks of authored DC ladder traces.

**`instruction_for_question` — L29**
```python
instruction_for_question(question: str, source: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `circuit._question`, `ValueError`
- 반환 표현: `f"Trace contract {VERSION}: supports only ideal DC ladders with 1–2 series sources and 2–3 resistor stages. Make the FIRST nonempty line after Solution: exactly Trace: followed by one J …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_rational` — L64**
```python
_rational(value: object) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `re.fullmatch`, `str`, `Fraction`
- 반환 표현: `True`; `False`; `str(Fraction(value)) == value`

**`_node` — L73**
```python
_node(value: object) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `re.fullmatch`
- 반환 표현: `type(value) is str and re.fullmatch('N[0-5]', value) is not None`

**`_resistor` — L77**
```python
_resistor(value: object) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `re.fullmatch`
- 반환 표현: `type(value) is str and re.fullmatch('R[1-6]', value) is not None`

**`schema` — L81**
```python
schema(trace: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `type`, `len`, `any`, `_node`, `_rational`, `_resistor`, `all`
- 반환 표현: `False`; `type(query) is list and len(query) == 4 and (type(query[0]) is str) and (query[0] in ('current', 'voltage', 'resistance')) and (_resistor(query[1]) if query[0] == 'current' else _node(q …`

**`inventory` — L117**
```python
inventory(trace: dict) -> list[tuple[str, str]]
```
- 기존 설명: Only syntactic regions, never native values or verdicts.
- 직접 호출: `schema`, `ValueError`, `rows.extend`, `range`, `len`
- 반환 표현: `[*rows, *_INPUT, ('CIRCUIT_QUERY_SPEC', 'query'), ('CIRCUIT_QUERY_VALUE', 'query')]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`identity` — L128**
```python
identity(trace: dict, rule: str, locus: str) -> list | None
```
- 기존 설명: Operation identity follows node/resistor/orientation, not a trace array index.
- 직접 호출: `schema`, `inventory`, `int`
- 반환 표현: `None`; `['circuit_ladder', rule, trace['nodes'][int(locus[6:-1])][0]]`; `['circuit_ladder', rule, *row[:3]]`; 외 3개 분기

**`supported_unknown` — L144**
```python
supported_unknown(row: dict, checks: list[dict]) -> bool
```
- 기존 설명: Only after fresh report validation; accepts intrinsic nulls and exact null dependencies.
- 직접 호출: `row.get`, `check.get`, `actual.get`, `all`
- 반환 표현: `False`; `actual.get((row.get('rule'), row.get('locus'))) == 'UNVERIFIABLE' and all((actual.get((ref['rule'], ref['locus'])) == 'UNVERIFIABLE' for ref in row.get('blocked_by', [])))`

**`_public_checks` — L154**
```python
_public_checks(trace: dict) -> dict[tuple[str, str], tuple[bool | None, list[dict]]]
```
- 기존 설명: Recompute all local arithmetic and exact null dependencies without a private world.
- 직접 호출: `enumerate`, `nodes.values`, `add`, `all`, `Fraction`, `sum`, `zip`, `ohm.get`, `nodes.get`
- 반환 표현: `checks`

**`_contract` — L218**
```python
_contract(actor: ActorItem, item: VerifierItem) -> tuple[dict, list[dict], dict, tuple[str, str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `json.dumps`, `type`, `world.get`, `ValueError`, `circuit._validate`, `any`, `{'current': 'A', 'voltage': 'V', 'resistance': 'ohm'}.get`, `circuit._facts`, `actor.problem_id.startswith` 외 5개
- 반환 표현: `(world, circuit._components(world), first, query_tuple)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`check_trace` — L245**
```python
check_trace(*, actor: ActorItem, item: VerifierItem, text: str, finish_reason: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_contract`, `add`, `_finish`, `_trace`, `schema`, `observation_spans`, `len`, `range`, `_response_scope` 외 12개
- 반환 표현: `_finish(report)`

**`validate_report` — L309**
```python
validate_report(report: dict, text: str, finish_reason: str) -> None
```
- 기존 설명: Bind raw evidence, full public inventory, arithmetic and exact null dependencies.
- 직접 호출: `len`, `json.dumps`, `set`, `_sha`, `type`, `ValueError`, `_trace`, `_finish`, `dict`, `schema` 외 4개
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.051 `fine_tuning/notebook_rl/cli.py`

[현재 소스](../../fine_tuning/notebook_rl/cli.py) · SHA256 `12a820ee71ec221aa03435c9b455de8954c010624246ac96386b02159dc05cce`

모듈의 기존 설명: CPU preparation commands. Generation and audits do not authorize GPU training.

**`artifact_path` — L16**
```python
artifact_path(path: Path) -> Path
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(__file__).resolve`, `Path`, `path.resolve().is_relative_to`, `path.resolve`, `any`, `p.is_symlink`, `ValueError`, `_safe_root`
- 반환 표현: `path`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`write_pool` — L25**
```python
write_pool(path: Path, rows: list[GeneratedProblem], *, recipe: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `artifact_path`, `len`, `ValueError`, `path.mkdir`, `write_split`, `set`, `zip`, `_write`, `written.add`, `load_images` 외 4개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`read_pool` — L46**
```python
read_pool(path: Path) -> list[GeneratedProblem]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_safe_root`, `json.loads`, `_read`, `info.get`, `_sha`, `ValueError`, `read_actors`, `read_verifier`, `read_lineage`, `len` 외 6개
- 반환 표현: `records`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`check_measurement_data` — L65**
```python
check_measurement_data(job_path: Path) -> dict
```
- 기존 설명: Open a public path/hash job, never its private index or answer payloads.
- 직접 호출: `json.loads`, `_read`, `type`, `set`, `isinstance`, `Path(job['build_root']).is_absolute`, `Path`, `ValueError`, `Path(value['path']).is_absolute`, `re.fullmatch` 외 2개
- 반환 표현: `result | {'actors': [json.loads(to_json(actor)) for actor in result['actors']]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L86**
```python
main(argv: list[str] | None=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_subparsers`, `commands.add_parser`, `generate.add_argument`, `build.add_argument`, `audit.add_argument`, `Path(__file__).with_name`, `Path`, `select.add_argument`, `check.add_argument` 외 22개
- 반환 표현: `2 if result.get('status') == 'BLOCKED_NOT_CHECKED' else 0`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

### D18.052 `fine_tuning/notebook_rl/contamination.py`

[현재 소스](../../fine_tuning/notebook_rl/contamination.py) · SHA256 `caf0d31785ae9e88948b7b1ef6ecf39e72e6fef023165c4d98a7e7eef51c91dd`

모듈의 기존 설명: Read-only exclusion scans, not dataset/training approval or contamination-free proof.

**`_json` — L55**
```python
_json(value: object) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ': …`, `json.dumps`
- 반환 표현: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()`

**`_digest` — L60**
```python
_digest(value: object) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(_json(value)).hexdigest`, `hashlib.sha256`, `_json`
- 반환 표현: `hashlib.sha256(_json(value)).hexdigest()`

**`normalize_text` — L64**
```python
normalize_text(text: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `unicodedata.normalize`, `re.sub`, `' '.join`, `value.casefold().split`, `value.casefold`
- 반환 표현: `' '.join(value.casefold().split())`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_tokens` — L73**
```python
_tokens(text: str) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `re.findall`, `normalize_text`
- 반환 표현: `re.findall('\\w+', normalize_text(text))`

**`_grams` — L77**
```python
_grams(text: str, n: int) -> set[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_tokens`, `' '.join`, `range`, `len`
- 반환 표현: `{' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}`

**클래스 `BKTree` — L82**

기저: `object`.

**`BKTree.__init__` — L84**
```python
BKTree.__init__(self) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`BKTree.add` — L87**
```python
BKTree.add(self, value: int, reference: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.nodes.append`, `(value ^ key).bit_count`, `refs.append`, `len`
- 반환 표현: `None`

**`BKTree.search` — L104**
```python
BKTree.search(self, value: int, radius: int=3) -> list[tuple[int, dict]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `pending.pop`, `(value ^ key).bit_count`, `found.extend`, `pending.extend`, `children.items`
- 반환 표현: `found`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_low_information` — L119**
```python
_low_information(value: str) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `set`, `bytes.fromhex`
- 반환 표현: `len(set(bytes.fromhex(value))) <= 1`

**`image_fingerprints` — L123**
```python
image_fingerprints(image: Image.Image) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `image.convert`, `min`, `ValueError`, `regions.update`, `range`, `np.cos`, `np.arange`, `regions.items`, `rgb.crop(box).convert`, `rgb.crop` 외 11개
- 반환 표현: `{'pixels_sha256': audit.image_fingerprint(rgb)['pixels_sha256'], 'variants': variants}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_coverage` — L146**
```python
_coverage(data: dict) -> tuple[bool, list[str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `data.get`, `type`, `coverage.get`, `ValueError`, `Counter`, `set`, `defaultdict`, `isinstance`, `entry.get`, `_SHA.fullmatch` 외 9개
- 반환 표현: `(complete, [] if complete else ['Incomplete native MMMU/Pro file/record coverage'])`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_pinned_coverage` — L196**
```python
_pinned_coverage(files: list[dict]) -> bool
```
- 기존 설명: Bind native shards, including unequal/sharded test subjects, to pinned bytes.
- 직접 호출: `vault._read`, `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`, `ValueError`, `json.loads`, `len`, `set`, `aliases.get`, `any`, `type` 외 3개
- 반환 표현: `False`; `True`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_text_cache` — L227**
```python
_text_cache(entries: list[dict]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `defaultdict`, `enumerate`, `normalize_text`, `exact[hashlib.sha256(text.encode()).hexdigest()].append`, `hashlib.sha256(text.encode()).hexdigest`, `hashlib.sha256`, `text.encode`, `_grams`, `inverted[n][gram].add`
- 반환 표현: `{'entries': entries, 'exact': exact, 'sets': sets, 'inverted': inverted}`

**`_prepare` — L243**
```python
_prepare(data: dict, partial_name: str | None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_coverage`, `data.get`, `isinstance`, `partial_name.strip`, `ValueError`, `BKTree`, `defaultdict`, `limits.extend`, `type`, `any` 외 33개
- 반환 표현: `{'data': data, 'sha256': _digest(data), 'full_coverage': full and (not historical) and (partial_name is None), 'partial_name': partial_name, 'limitations': limits, 'text_cache': cache,  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_index` — L344**
```python
load_index(path: Path, *, partial_name: str | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `vault._read`, `Path`, `json.loads`, `_prepare`, `str`, `Path(path).absolute`, `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`
- 반환 표현: `prepared`

**`_text_matches` — L352**
```python
_text_matches(text: str, field: str, cache: dict) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `normalize_text`, `_tokens`, `hashlib.sha256(normal.encode()).hexdigest`, `hashlib.sha256`, `normal.encode`, `len`, `cache.get`, `set`, `result.append`, `cache['exact'].get` 외 9개
- 반환 표현: `result`

**`cosine_similarities` — L381**
```python
cosine_similarities(queries, references) -> np.ndarray
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `np.asarray`, `np.isfinite(q).all`, `np.isfinite`, `np.isfinite(r).all`, `ValueError`, `np.linalg.norm`, `np.isfinite(qn).all`, `np.isfinite(rn).all`, `np.any`, `np.clip`
- 반환 표현: `np.clip(q / qn[:, None] @ (r / rn[:, None]).T, -1, 1)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`family_quarantine` — L392**
```python
family_quarantine(items: dict[str, dict], candidates: list[dict]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `items.items`, `item.get`, `root`, `sorted`, `set`
- 반환 표현: `{'quarantined_ids': quarantined, 'allowlist': sorted(set(items) - set(quarantined)), 'families': sorted({items[pid]['structure_hash'] for pid in quarantined})}`

**`_source_checks` — L413**
```python
_source_checks(actor, lineage, registry: dict) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `registry.get`, `ValueError`, `set`, `type`, `any`, `isinstance`, `pending.pop`, `seen.add`, `sources.get`, `entry.keys` 외 5개
- 반환 표현: `sorted(set(reasons))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_snapshot` — L459**
```python
_snapshot(split_roots: dict[str, Path], image_root: Path) -> tuple[dict, dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `vault._safe_root`, `sorted`, `split_roots.items`, `ValueError`, `str`, `hashlib.sha256(vault._read(root / name)).hexdigest`, `hashlib.sha256`, `vault._read`, `vault.read_actors`, `vault.read_verifier` 외 21개
- 반환 표현: `(items, files, images)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_calibration` — L514**
```python
_calibration(evidence: dict, registry: dict, index_sha: str, protocol_sha: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `evidence.get`, `registry.get`, `type`, `policy.get`, `_digest`, `calibration.get`, `ValueError`, `math.isfinite`, `set`, `engines.values` 외 5개
- 반환 표현: `thresholds`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_engine` — L551**
```python
_engine(stage: dict, name: str, evidence: dict, registry: dict, index_sha: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `registry['scanner_policy'].get('engines', {}).get`, `registry['scanner_policy'].get`, `type`, `any`, `stage.get`, `policy.get`, `isinstance`, `_SHA.fullmatch`, `_digest`, `ValueError` 외 1개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`ocr_scan` — L571**
```python
ocr_scan(stage: dict, index: dict, items: dict, images: dict) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `stage.get`, `items.values`, `type`, `set`, `ValueError`, `group.items`, `entry.get`, `isinstance`, `entry['text'].strip`, `benchmark.items` 외 6개
- 반환 표현: `found`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_vectors` — L599**
```python
_vectors(entries: object, keys: set[str], bindings: dict[str, str]) -> tuple[list[str], list[list[float]]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `ValueError`, `sorted`, `entry.get`, `any`, `math.isfinite`, `vectors.append`
- 반환 표현: `(ordered, vectors)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`embed_scan` — L613**
```python
embed_scan(stage: dict, kind: str, index: dict, items: dict, images: dict, thresholds: dict) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_vectors`, `stage.get`, `set`, `hashlib.sha256(normalize_text(entry['text']).encode()).hexdigest`, `hashlib.sha256`, `normalize_text(entry['text']).encode`, `normalize_text`, `items.items`, `ValueError`, `jobs.extend` 외 11개
- 반환 표현: `found`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`scan` — L649**
```python
scan(split_roots: dict[str, Path], image_root: Path, index: dict, source_registry: dict, evidence: dict | None=None) -> dict
```
- 기존 설명: Collect real candidates even in blocked/partial scans; never silently skip a stage.
- 직접 호출: `index.get`, `_prepare`, `json.loads`, `_json`, `hashlib.sha256(vault._read(Path(index['path']))).hexdigest`, `hashlib.sha256`, `vault._read`, `Path`, `ValueError`, `str` 외 50개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`write_receipt` — L775**
```python
write_receipt(path: Path, receipt: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `receipt.items`, `receipt.get`, `_digest`, `ValueError`, `vault._safe_root`, `Path`, `_json`, `Path(path).open`, `os.fchmod`, `stream.fileno` 외 5개
- 반환 표현: `{'path': str(Path(path).absolute()), 'sha256': hashlib.sha256(raw).hexdigest()}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.053 `fine_tuning/notebook_rl/dataset.py`

[현재 소스](../../fine_tuning/notebook_rl/dataset.py) · SHA256 `4f988cae5ae9c93f7b2a1ab687064828824634bc457eac63de756d4b883a6a2a`

모듈의 기존 설명: Write-once candidate builds and pre-learning selection. Preparation is not approval.

**`_bytes` — L24**
```python
_bytes(value: Any) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=Fal …`, `json.dumps`
- 반환 표현: `json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()`

**`_sha` — L28**
```python
_sha(value: bytes) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(value).hexdigest`, `hashlib.sha256`
- 반환 표현: `hashlib.sha256(value).hexdigest()`

**`_write` — L32**
```python
_write(path: Path, payload: bytes) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.parent.mkdir`, `path.open`, `os.fchmod`, `stream.fileno`, `stream.write`, `stream.flush`, `os.fsync`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`generate_pool` — L41**
```python
generate_pool(generator: str, n: int, seed: int, split: Split, mode_mix: float=0.5) -> list[GeneratedProblem]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `math.isfinite`, `ValueError`, `get_generator`, `Path(module.__file__).read_bytes`, `Path`, `Path(__file__).with_name('generators').joinpath('common.py').read_bytes`, `Path(__file__).with_name('generators').joinpath`, `Path(__file__).with_name`, `b''.join` 외 16개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_images` — L75**
```python
load_images(actor: ActorItem, image_root: Path) -> list[Image.Image]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_actor`, `_safe_root`, `zip`, `any`, `p.is_symlink`, `path.resolve().is_relative_to`, `path.resolve`, `root.resolve`, `ValueError`, `_read` 외 11개
- 반환 표현: `images`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`build_splits` — L100**
```python
build_splits(root: Path, pools: dict[str, list[GeneratedProblem]], *, index: dict | None=None, source_registry: dict | None=None, evidence: dict | None=None) -> dict
```
- 기존 설명: Candidate generation/solver checks never assert human review or training permission.
- 직접 호출: `set`, `any`, `pools.values`, `ValueError`, `pools.items`, `to_json`, `seen_ids.add`, `sources.add`, `len`, `zip` 외 16개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`freeze_selection` — L168**
```python
freeze_selection(phase_a_report: Path, rules: dict, *, output: Path) -> dict
```
- 기존 설명: No answers/teacher targets here: only pre-learning rollout counts select difficulty.
- 직접 호출: `set`, `any`, `type`, `ValueError`, `_read`, `json.loads`, `report.get`, `_sha`, `_bytes`, `sorted` 외 14개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.054 `fine_tuning/notebook_rl/engines.py`

[현재 소스](../../fine_tuning/notebook_rl/engines.py) · SHA256 `035aa4bbf9d5c342b0fa4727617a6db74e02d168896dd86686b28179ec1dc8da`

모듈의 기존 설명: Token-faithful rollout engines. vLLM operation still requires actual GPU gates.

**클래스 `SamplingSettings` — L24**

기저: `object`.

```python
temperature: float = 1.0
top_p: float = 1.0
top_k: int = 0
presence_penalty: float = 0.0
repetition_penalty: float = 1.0
```

**`SamplingSettings.validate` — L31**
```python
SamplingSettings.validate(self) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `type`, `math.isfinite`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`SamplingSettings.name` — L40**
```python
SamplingSettings.name(self) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.validate`
- 반환 표현: `'temperature_1' if self.temperature == 1 else 'qwen_official'`

**클래스 `RolloutRequest` — L49**

기저: `object`.

```python
request_id: str
prompt_token_ids: list[int] | None
messages: list[dict] | None
images: list
n: int
max_new_tokens: int
seeds: list[int]
```

**클래스 `Completion` — L60**

기저: `object`.

```python
token_ids: list[int]
logprobs: list[float]
finish_reason: str
text: str
```

**클래스 `RolloutResult` — L68**

기저: `object`.

```python
request_id: str
prompt_token_ids: list[int]
completions: list[Completion]
```

**클래스 `Prepared` — L75**

기저: `object`.

```python
token_ids: list[int]
pixels: dict
images: list
image_key: str
image_spans: tuple[tuple[int, int], ...] = ()
```

**`_request` — L83**
```python
_request(request: RolloutRequest) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `len`, `any`, `set`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_completion` — L95**
```python
_completion(ids: list[int], logprobs: list[float], finish: str, text: str) -> Completion
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `any`, `type`, `math.isfinite`, `RuntimeError`, `Completion`
- 반환 표현: `Completion(ids, logprobs, finish, text)`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**클래스 `InputPreparation` — L103**

기저: `object`.

**`InputPreparation.__init__` — L104**
```python
InputPreparation.__init__(self, processor: Any, model_config: Any, image_cfg: dict, context_limit: int) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dict`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`InputPreparation.initialize_weights` — L116**
```python
InputPreparation.initialize_weights(self, named_weights: Iterable[tuple[str, torch.Tensor]], policy_version: int) -> None
```
- 기존 설명: Restore a fresh engine once; actual single/multi-image probes must follow.
- 직접 호출: `type`, `ValueError`, `RuntimeError`, `engine._load_weights`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`InputPreparation.validate_sync` — L128**
```python
InputPreparation.validate_sync(self, probes: list[tuple[RolloutRequest, list[int], list[float]]], threshold: float) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `RuntimeError`, `len`, `sorted`, `type`, `ValueError`, `engine.probe_logprobs`, `abs`, `zip`, `any`, `math.isfinite` 외 3개
- 반환 표현: `reports`
- 명시적 예외: `RuntimeError`, `ValueError`; 전파되는 callee 예외는 별도.

**`InputPreparation.prepare` — L150**
```python
InputPreparation.prepare(self, request: RolloutRequest) -> Prepared
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_request`, `ValueError`, `hashes.append`, `hashlib.sha256(image.tobytes()).hexdigest`, `hashlib.sha256`, `image.tobytes`, `hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest`, `json.dumps(hashes, separators=(',', ':')).encode`, `json.dumps`, `prepare_protocol_inputs` 외 13개
- 반환 표현: `Prepared(ids, pixels, self._processed_images if request.images else [], key, tuple(spans))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`InputPreparation.model_inputs` — L204**
```python
InputPreparation.model_inputs(self, request: RolloutRequest, completion_ids: list[int], device: torch.device) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.prepare`, `len`, `ValueError`, `torch.tensor`, `torch.ones_like`, `v.to`, `prepared.pixels.items`
- 반환 표현: `{'input_ids': ids, 'attention_mask': torch.ones_like(ids), **{k: v.to(device) for k, v in prepared.pixels.items()}}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `_SelectedTokenLogprobs` — L213**

기저: `object`.

**`_SelectedTokenLogprobs.__init__` — L216**
```python
_SelectedTokenLogprobs.__init__(self, prompt_length: int) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_SelectedTokenLogprobs._append` — L222**
```python
_SelectedTokenLogprobs._append(self, token: torch.Tensor) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `RuntimeError`, `self.logprobs.append`, `self.scores[0].float().log_softmax(-1).gather`, `self.scores[0].float().log_softmax`, `self.scores[0].float`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`_SelectedTokenLogprobs.__call__` — L228**
```python
_SelectedTokenLogprobs.__call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `int`, `RuntimeError`, `self._append`
- 반환 표현: `scores`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`_SelectedTokenLogprobs.finish` — L238**
```python
_SelectedTokenLogprobs.finish(self, sequences: torch.Tensor) -> list[float]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `torch.equal`, `RuntimeError`, `self._append`, `torch.cat(self.logprobs).tolist`, `torch.cat`
- 반환 표현: `torch.cat(self.logprobs).tolist()`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`_SelectedTokenLogprobs.clear` — L247**
```python
_SelectedTokenLogprobs.clear(self) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.logprobs.clear`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**클래스 `HFRolloutEngine` — L252**

기저: `InputPreparation`.

**`HFRolloutEngine.__init__` — L255**
```python
HFRolloutEngine.__init__(self, model: Any, processor: Any, image_cfg: dict, max_model_len: int) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `super().__init__`, `super`, `_weight_inventory`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`HFRolloutEngine.sample` — L260**
```python
HFRolloutEngine.sample(self, requests: list[RolloutRequest], *, on_completion: Callable[[str, int, int, Completion], None] | None=None) -> list[RolloutResult]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.sample_with_settings`, `SamplingSettings`
- 반환 표현: `self.sample_with_settings(requests, SamplingSettings())`; `self.sample_with_settings(requests, SamplingSettings(), on_completion=on_completion)`

**`HFRolloutEngine.sample_with_settings` — L266**
```python
HFRolloutEngine.sample_with_settings(self, requests: list[RolloutRequest], sampling: SamplingSettings, *, on_completion: Callable[[str, int, int, Completion], None] | None=None) -> list[RolloutResult]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `RuntimeError`, `sampling.validate`, `NotImplementedError`, `len`, `ValueError`, `next`, `self.model.parameters`, `self.model.eval`, `self.prepare`, `self.model_inputs` 외 22개
- 반환 표현: `results`
- 명시적 예외: `RuntimeError`, `NotImplementedError`, `ValueError`; 전파되는 callee 예외는 별도.

**`HFRolloutEngine.probe_logprobs` — L327**
```python
HFRolloutEngine.probe_logprobs(self, request: RolloutRequest, completion_ids: list[int]) -> list[float]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `next`, `self.model.parameters`, `self.model_inputs`, `self.model.eval`, `torch.no_grad`, `self.model(**batch, use_cache=False).logits[:, -len(completion_ids) - 1:-1].float`, `self.model`, `len`, `logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1))[0, :, 0].tolist` 외 4개
- 반환 표현: `logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1))[0, :, 0].tolist()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`HFRolloutEngine.push_weights` — L342**
```python
HFRolloutEngine.push_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `self._load_weights`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`HFRolloutEngine._load_weights` — L348**
```python
HFRolloutEngine._load_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dict`, `self.model.named_parameters`, `tuple`, `parameters.items`, `ValueError`, `isinstance`, `bool`, `torch.isfinite(value).all`, `torch.isfinite`, `weights.keys` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`vllm_kwargs` — L368**
```python
vllm_kwargs(model_path: Path, image_cfg: dict, kv_cache_memory_bytes: int, max_num_seqs: int, max_model_len: int) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `type`, `image_cfg.get`, `ValueError`, `dict`, `str`
- 반환 표현: `dict(model=str(model_path), tokenizer=str(model_path), dtype='bfloat16', quantization=None, gpu_memory_utilization=0.7, enforce_eager=False, enable_prefix_caching=False, mm_processor_ca …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_weight_inventory` — L384**
```python
_weight_inventory(config: Any) -> dict[str, tuple[int, ...]]
```
- 기존 설명: The real HF architecture on meta, not a second allocation of the model weights.
- 직접 호출: `ValueError`, `torch.device`, `Qwen3VLForConditionalGeneration`, `tuple`, `model.named_parameters`
- 반환 표현: `{name: tuple(p.shape) for name, p in model.named_parameters()}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_clear_encoder_cache` — L395**
```python
_clear_encoder_cache(worker: Any) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `RuntimeError`, `cache.clear`
- 반환 표현: `not cache`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**클래스 `VLLMRolloutEngine` — L404**

기저: `InputPreparation`.

**`VLLMRolloutEngine.__init__` — L406**
```python
VLLMRolloutEngine.__init__(self, model_path: Path, image_cfg: dict, kv_cache_memory_bytes: int, max_num_seqs: int, max_model_len: int) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `version`, `RuntimeError`, `import_module`, `AutoProcessor.from_pretrained`, `str`, `super().__init__`, `super`, `AutoConfig.from_pretrained`, `_weight_inventory`, `Path(model_path).resolve` 외 3개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`VLLMRolloutEngine._prompt` — L428**
```python
VLLMRolloutEngine._prompt(self, prepared: Prepared, ids: list[int]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `ValueError`, `transport.extend`, `transport.append`, `prompt.update`, `range`
- 반환 표현: `prompt`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`VLLMRolloutEngine.sample` — L451**
```python
VLLMRolloutEngine.sample(self, requests: list[RolloutRequest]) -> list[RolloutResult]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.sample_with_settings`, `SamplingSettings`
- 반환 표현: `self.sample_with_settings(requests, SamplingSettings())`

**`VLLMRolloutEngine.sample_with_settings` — L454**
```python
VLLMRolloutEngine.sample_with_settings(self, requests: list[RolloutRequest], sampling: SamplingSettings) -> list[RolloutResult]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `import_module`, `sampling.validate`, `RuntimeError`, `len`, `ValueError`, `self.prepare`, `zip`, `prompts.append`, `self._prompt`, `params.append` 외 11개
- 반환 표현: `results`
- 명시적 예외: `RuntimeError`, `ValueError`; 전파되는 callee 예외는 별도.

**`VLLMRolloutEngine.probe_logprobs` — L492**
```python
VLLMRolloutEngine.probe_logprobs(self, request: RolloutRequest, completion_ids: list[int]) -> list[float]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `import_module`, `self.prepare`, `len`, `ValueError`, `self.llm.generate`, `self._prompt`, `SamplingParams`, `list`, `RuntimeError`, `float` 외 1개
- 반환 표현: `[float(row[t].logprob) for row, t in zip(rows, completion_ids, strict=True)]`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`VLLMRolloutEngine.push_weights` — L508**
```python
VLLMRolloutEngine.push_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `self._load_weights`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`VLLMRolloutEngine._load_weights` — L514**
```python
VLLMRolloutEngine._load_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `import_module`, `isinstance`, `RuntimeError`, `scheduler.has_unfinished_requests`, `any`, `manager.cached.values`, `set`, `self.llm.apply_model`, `self._weight_shapes.keys`, `scheduler.reset_prefix_cache` 외 3개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

### D18.055 `fine_tuning/notebook_rl/episode_notebook.py`

[현재 소스](../../fine_tuning/notebook_rl/episode_notebook.py) · SHA256 `84326106ee88fbc21d42a10ea92981bb3da010d9ef323f07a0885d1f02367084`

모듈의 기존 설명: All-attempt notebook actions; private verification, not self-reported routing.

**`solution_samples` — L141**
```python
solution_samples(tree: ProblemTree) -> list[Sample]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `[s for g in tree.groups if g.kind != 'DIAG' for s in g.samples]`

**`solution_hash` — L145**
```python
solution_hash(tree: ProblemTree) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_hash`, `json.loads`, `to_json`, `solution_skips`
- 반환 표현: `_hash({'groups': [json.loads(to_json(g)) for g in tree.groups if g.kind != 'DIAG'], 'skipped': solution_skips(tree)})`

**`solution_skips` — L150**
```python
solution_skips(tree: ProblemTree) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tree.notebook.get`, `row.get`
- 반환 표현: `[row for row in tree.notebook.get('skipped_branches', []) if row.get('kind') != 'DIAG']`

**`public_transcript` — L154**
```python
public_transcript(samples: list[Sample]) -> list[dict]
```
- 기존 설명: Contains raw model output and public outcome only, never verifier slot values.
- 직접 호출: `enumerate`, `ids.get`, `_first_attempt_lines`
- 반환 표현: `[{'attempt': ids[s.sample_id], 'parent': ids.get(s.parent_sample_id) if s.parent_sample_id else None, 'turn': s.turn, 'status': s.result.status if s.result else 'SYSTEM_ERROR', 'finish_ …`

**`strict_routing` — L163**
```python
strict_routing(tree: ProblemTree) -> dict
```
- 기존 설명: Derive admission from bound verifier slots, never note or audit claims.
- 직접 호출: `solution_samples`, `enumerate`, `verified_wrong_slots`, `visual.append`, `required_observations_verified`, `complete.append`, `bool`, `len`, `solution_skips`, `all` 외 1개
- 반환 표현: `{'routing_rule': STRICT_ROUTE_RULE, 'admission': admission, 'attempt_count': len(samples), 'complete_episode': bool(samples) and (not solution_skips(tree)) and all((g.usable for g in tr …`

**`verify_replayed_outcome` — L189**
```python
verify_replayed_outcome(stored: VerifierResult, fresh: VerifierResult, *, replay_relation_v1: bool=False) -> None
```
- 기존 설명: Allow one historical version transition only after exact semantic regrading.
- 직접 호출: `ValueError`, `replace`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`audit_episode` — L204**
```python
audit_episode(tree: ProblemTree, actor: ActorItem, verifier: Any, *, routing_rule: str='all-observed-required-slots-v1', replay_relation_v1: bool=False, evaluation_only: bool=False) -> dict
```
- 기존 설명: Freshly verify every actual attempt before constructing a model prompt.
- 직접 호출: `to_json`, `type`, `ValueError`, `len`, `tree.notebook.get`, `any`, `_hash`, `enumerate`, `solution_samples`, `solution_skips` 외 17개
- 반환 표현: `audit`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`public_observation_differences` — L282**
```python
public_observation_differences(samples: list[Sample]) -> list[dict]
```
- 기존 설명: Index differing model-written values, without a verifier or truth inference.
- 직접 호출: `enumerate`, `_first_attempt_lines`, `observation_spans`, `len`, `ValueError`, `by_slot.setdefault(key, []).append`, `by_slot.setdefault`, `sorted`, `by_slot.items`
- 반환 표현: `[{'slot': key, 'observations': rows} for key, rows in sorted(by_slot.items()) if len({row['value'] for row in rows}) > 1]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`instruction` — L296**
```python
instruction(samples: list[Sample], audit: dict, observation_instruction: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.items`, `len`, `public_transcript`, `json.dumps`, `transcript.replace('<', '\\u003c').replace`, `transcript.replace`, `public_observation_differences`, `differences.replace('<', '\\u003c').replace`, `differences.replace`, `PROMPT.format`
- 반환 표현: `PROMPT.format(observation_instruction=observation_instruction, success_attempt=audit['success_attempt'], transcript=transcript, observation_differences=differences)`

**`_evidence_line_ids` — L313**
```python
_evidence_line_ids(ranges: Any, line_count: int) -> list[int] | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `len`, `any`, `selected.extend`, `range`
- 반환 표현: `None`; `selected`

**`_fact_projection` — L331**
```python
_fact_projection(facts: dict, answer: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `'\n'.join`, `facts.items`
- 반환 표현: `'Observed facts:\n' + '\n'.join((f'{k} = {v}' for k, v in facts.items())) + '\nSolution:\nFinal answer: ' + answer`

**`_reinspection_slots` — L336**
```python
_reinspection_slots(facts: dict, result: VerifierResult | None) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `slot_id`, `set`, `verified_wrong_slots`
- 반환 표현: `sorted({slot_id(k) for k in facts} & (set(verified_wrong_slots(slots)) | set(slots.conflicting))) if slots else []`

**`grade_note` — L342**
```python
grade_note(text: str, finish_reason: str, tree: ProblemTree, audit: dict, verifier: Any, *, feedback_policy: str | None=None, evaluation_only: bool=False, actor: ActorItem | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `audit_episode`, `audit.get`, `any`, `solution_samples`, `solution_hash`, `record.update`, `json.loads`, `set` 외 23개
- 반환 표현: `record`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_feedback_binding` — L449**
```python
_validate_feedback_binding(checked: dict, sample: Sample) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `checked.get`, `type`, `ValueError`, `any`, `slot_id`, `sorted`, `set`, `to_json`, `json.loads`, `_validate` 외 1개
- 반환 표현: `[]`; `hint`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`notebook_feedback` — L475**
```python
notebook_feedback(checked: dict, sample: Sample | None=None) -> str
```
- 기존 설명: Expose check outcomes, never private targets, expected causes or raw evidence.
- 직접 호출: `checked.get`, `synthesis.notebook_feedback`, `type`, `ValueError`, `set`, `any`, `checks.values`, `row.get`, `len`, `attempts.append` 외 4개
- 반환 표현: `synthesis.notebook_feedback(checked, sample)`; `instruction + json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_update_binding` — L510**
```python
validate_update_binding(tree: ProblemTree, *, tokenizer: Any | None=None, generation_policy: Any=_UNSPECIFIED_FEEDBACK_POLICY, feedback_policy: Any=_UNSPECIFIED_FEEDBACK_POLICY, observation_policy: Any=_UNSPECIFIED_FEEDBACK_POLICY, draft_policy: Any=_UNSPECIFIED_FEEDBACK_POLICY, preparation: Any | None=None, images: list | None=None, observation_actor: ActorItem | None=None, observation_schema: str | None=None, observation_master_seed: int | None=None) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tree.notebook.get`, `note.get`, `whole_reference_page_rollout.validate`, `ValueError`, `pair_action_notebook.validate_update_binding`, `_hash`, `drafts.contract`, `type`, `any`, `reference.contract` 외 58개
- 반환 표현: `whole_reference_page_rollout.validate(tree, tokenizer=tokenizer, generation_policy=generation_policy, feedback_policy=feedback_policy, observation_policy=observation_policy, draft_polic …`; `pair_action_notebook.validate_update_binding(tree, tokenizer=tokenizer, generation_policy=generation_policy, feedback_policy=feedback_policy, observation_policy=observation_policy, prep …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_partial_stop_reason` — L824**
```python
_partial_stop_reason(artifacts: list[dict]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `all`
- 반환 표현: `'PARTIAL_NOTEBOOK_CAPTURED'`; `'NO_NOTEBOOK_CLAIMS' if statuses and all((status == 'NO_NOTEBOOK_CLAIMS' for status in statuses)) else 'PARTIAL_NOTEBOOK_CAPTURE_FAILED'`

**`append_episode_notebooks` — L832**
```python
append_episode_notebooks(trees: list[ProblemTree], images: dict[str, list], engine: Any, verifier: Any, prompts: dict, cfg: RolloutConfig, *, actors: dict[str, ActorItem], on_group: Callable[[Group], None] | None=None, on_observation: Callable[[dict], None] | None=None, on_draft: Callable[[dict], None] | None=None, on_page: Callable[[dict], None] | None=None, should_stop: Callable[[], bool] | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `prompts.get`, `whole_reference_page_rollout.append`, `ValueError`, `pair_action_notebook.append`, `cfg.validate`, `_hash`, `drafts.contract`, `type`, `any`, `reference.contract` 외 61개
- 반환 표현: `whole_reference_page_rollout.append(trees, images, engine, verifier, prompts, cfg, actors=actors, on_group=on_group, on_observation=on_observation, on_draft=on_draft, on_page=on_page, s …`; `pair_action_notebook.append(trees, images, engine, verifier, prompts, cfg, actors=actors, on_group=on_group, on_observation=on_observation, should_stop=should_stop)`; `report`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`regrade_synthesis` — L1123**
```python
regrade_synthesis(tree: ProblemTree, actor: ActorItem, verifier: Any, tokenizer: Any) -> dict
```
- 기존 설명: Recheck actual action tokens, complete history and every note before commitment.
- 직접 호출: `tree.notebook.get('rewarded_notebook', {}).get`, `tree.notebook.get`, `whole_reference_page_rollout.regrade`, `ValueError`, `validate_update_binding`, `audit_episode`, `synthesis.audit_episode`, `tokenizer.decode`, `pair_action_notebook._grade_note_audited`, `grade` 외 3개
- 반환 표현: `whole_reference_page_rollout.regrade(tree, actor, verifier, tokenizer)`; `receipt | {'sha256': _hash(receipt)}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.056 `fine_tuning/notebook_rl/evaluate_external.py`

[현재 소스](../../fine_tuning/notebook_rl/evaluate_external.py) · SHA256 `bdfc5098a790859cb7bde7b339250dbd4e18ae9cb98e4f5d61c2cc65f85862d3`

모듈의 기존 설명: Gold-isolated external evaluation; CPU adapters test contracts, not GPU accuracy.

**`_sha` — L36**
```python
_sha(path: Path) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(path.read_bytes()).hexdigest`, `hashlib.sha256`, `path.read_bytes`
- 반환 표현: `hashlib.sha256(path.read_bytes()).hexdigest()`

**`_seed` — L40**
```python
_seed(problem_id: str, run_seed: int, sampler: str, index: int, turn: int=1) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int.from_bytes`, `hashlib.sha256(value.encode()).digest`, `hashlib.sha256`, `value.encode`
- 반환 표현: `int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big') % 2 ** 31`

**`pass_at_k` — L45**
```python
pass_at_k(n: int, correct: int, k: int) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `type`, `ValueError`, `math.comb`
- 반환 표현: `1.0 - (math.comb(n - correct, k) / math.comb(n, k) if n - correct >= k else 0.0)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_item_mean` — L51**
```python
_item_mean(rows: list[dict], value) -> tuple[float | None, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `defaultdict`, `value`, `items[row['problem_id']].append`, `float`, `statistics.mean`, `items.values`, `len`
- 반환 표현: `(statistics.mean((statistics.mean(v) for v in items.values())), len(items)) if items else (None, 0)`

**`_cell` — L60**
```python
_cell(rows: list[dict], k_list: tuple[int, ...]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `defaultdict`, `trials[row['problem_id'], row['run_seed'], row['parent_record_id']].append`, `sum`, `trials.values`, `type`, `len`, `all`, `_item_mean`, `sorted`, `statistics.median` 외 7개
- 반환 표현: `result`

**`summarize` — L118**
```python
summarize(rows: list[dict], k_list: tuple[int, ...]=(1, 4, 8)) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `set`, `any`, `type`, `ValueError`, `defaultdict`, `cells[row['model'], row['split'], family, row['mode'], row['sampler'], prompt_mode, answer …`, `sorted`, `cells.items`, `output.append` 외 4개
- 반환 표현: `{'status': 'COMPLETE' if all((r['status'] == 'COMPLETE' for r in output)) else 'INCOMPLETE', 'n_records': len(rows), 'n_items': len({r['problem_id'] for r in rows}), 'cells': output, 'p …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`evaluate` — L145**
```python
evaluate(model_dir: Path, split_root: Path, output: Path, modes, k_list=(1, 4, 8), seeds=(3407,), *, engine: Any, protocol: dict, image_root: Path | None=None, split: str | None=None, forbidden_roots=(), _test_verifier: Any=None) -> dict
```
- 기존 설명: Write every requested/failed observation; summary is the final completion marker.
- 직접 호출: `Path`, `tuple`, `len`, `set`, `any`, `type`, `ValueError`, `load_prompts`, `Path(__file__).with_name`, `bool` 외 34개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.057 `fine_tuning/notebook_rl/exclusion_bundle.py`

[현재 소스](../../fine_tuning/notebook_rl/exclusion_bundle.py) · SHA256 `87f2106a8b80f252b1850cab5b7fad8cda600d25cb8992b2d94f2ffc76fb7e10`

모듈의 기존 설명: Join complete private native input and real frozen detector caches; never approve training.

**`read_json` — L16**
```python
read_json(path: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `vault._read`
- 반환 표현: `json.loads(vault._read(path), object_pairs_hook=vault._unique)`

**`text_entries` — L20**
```python
text_entries(data: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `record['texts'].items`, `enumerate`, `c.normalize_text`, `c._digest`
- 반환 표현: `entries`

**`prepare_manifests` — L33**
```python
prepare_manifests(index_path: Path, raw_images: Path, build: Path, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read_json`, `c._coverage`, `data.get`, `ValueError`, `c._snapshot`, `str`, `(raw_images / sha).absolute`, `paths.update`, `files.items`, `paths.items` 외 16개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`cache_payloads` — L57**
```python
cache_payloads(root: Path, keys: set[str], *, texts=False) -> tuple[dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read_json`, `identity.get`, `hashlib.sha256(vault._read(Path(__file__).with_name('exclusion_engines.py'))).hexdigest`, `hashlib.sha256`, `vault._read`, `Path(__file__).with_name`, `Path`, `hashlib.sha256(vault._read(Path(c.__file__))).hexdigest`, `ValueError`, `c._digest` 외 3개
- 반환 표현: `(identity, result)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`assemble` — L77**
```python
assemble(index_path: Path, build: Path, calibration_path: Path, cache_roots: dict[str, Path], source_registry_path: Path, output: Path) -> dict
```
- 기존 설명: Full scan emits all unresolved candidates. This never silently releases them.
- 직접 호출: `c.load_index`, `read_json`, `calibration.get`, `ValueError`, `c._snapshot`, `items.values`, `set`, `cache_payloads`, `calibration.get('engines', {}).get`, `c._digest` 외 23개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L142**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `prepare_manifests`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.058 `fine_tuning/notebook_rl/exclusion_calibration.py`

[현재 소스](../../fine_tuning/notebook_rl/exclusion_calibration.py) · SHA256 `b0840e93d9e9e1d3c1b225c36c1597188a4241ab87b07ad3349873712ffb5f64`

모듈의 기존 설명: Private, deterministic copy-detector calibration material, never training data.

**`transform` — L30**
```python
transform(raw: bytes, name: str) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `Image.open`, `io.BytesIO`, `original.convert`, `image.save`, `output.getvalue`, `image.resize`, `max`, `image.crop`, `ImageOps.expand` 외 1개
- 반환 표현: `output.getvalue()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_image` — L52**
```python
_image(path: Path, sha: str) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `vault._read`, `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`, `ValueError`
- 반환 표현: `raw`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare` — L59**
```python
prepare(index_path: Path, raw_images: Path, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `c.load_index`, `ValueError`, `_private_root`, `vault._safe_root`, `set`, `source_sets['val'].update`, `source_sets['pro'].update`, `sorted`, `len`, `_image` 외 15개
- 반환 표현: `{'material': str(output / 'material.json'), 'positives': len(positives), 'negatives': len(negatives), 'images': len(images), 'training_allowed': False}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`threshold_for_fpr` — L130**
```python
threshold_for_fpr(scores: list[float], target: float=0.01) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `math.isfinite`, `ValueError`, `sorted`, `math.floor`, `len`, `max`, `math.nextafter`
- 반환 표현: `threshold`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_rate` — L142**
```python
_rate(positive: list[float], negative: list[float], threshold: float, examples: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `len`, `sum`, `c._digest`
- 반환 표현: `{'positive_count': len(positive), 'negative_count': len(negative), 'tpr': sum((v >= threshold for v in positive)) / len(positive), 'fpr': sum((v >= threshold for v in negative)) / len(n …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`measure` — L151**
```python
measure(material_path: Path, sscd: dict, siglip: dict, output: Path, *, engines: dict) -> dict
```
- 기존 설명: Vectors are actual cache exports SHA-bound to every calibration image.
- 직접 호출: `vault._read`, `json.loads`, `data.get`, `ValueError`, `hashlib.sha256(vault._read(Path(__file__))).hexdigest`, `hashlib.sha256`, `Path`, `hashlib.sha256(vault._read(Path(c.__file__))).hexdigest`, `set`, `engines['sscd'].get` 외 30개
- 반환 표현: `calibration`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L235**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `prepare`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.059 `fine_tuning/notebook_rl/exclusion_engines.py`

[현재 소스](../../fine_tuning/notebook_rl/exclusion_engines.py) · SHA256 `578c3fda23f8146915bfaadfad79ae32bfd41290af6132d92449f5c8aaaf9dd2`

모듈의 기존 설명: Resumable private OCR/SSCD/SigLIP caches from fixed local artifacts only.

**`_sha` — L35**
```python
_sha(raw: bytes) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`
- 반환 표현: `hashlib.sha256(raw).hexdigest()`

**`_image` — L39**
```python
_image(path: Path, expected: str) -> Image.Image
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.is_absolute`, `ValueError`, `vault._read`, `_sha`, `Image.open`, `io.BytesIO`, `image.load`, `image.convert`
- 반환 표현: `image.convert('RGB')`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_identity` — L50**
```python
_identity(name: str, revision: str, artifacts: dict[str, Path], preprocessing: dict, packages: tuple[str, ...]=()) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `audit.sha`, `path.resolve`, `sorted`, `artifacts.items`, `c._digest`, `importlib.metadata.version`
- 반환 표현: `{'engine_id': name, 'revision': revision, 'artifact_sha256': c._digest(hashes), 'artifact_files': hashes, 'preprocessing': preprocessing, 'producer_sha256': audit.sha(__file__), 'normal …`

**클래스 `OCR` — L59**

기저: `object`.

**`OCR.__init__` — L60**
```python
OCR.__init__(self, extracted: Path, timeout: float=60) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(extracted).resolve`, `Path`, `ValueError`, `str`, `subprocess.run`, `(version.stdout + version.stderr).decode().startswith`, `(version.stdout + version.stderr).decode`, `artifacts.update`, `p.relative_to`, `(self.root / 'usr/lib/x86_64-linux-gnu').glob` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`OCR.recognize` — L80**
```python
OCR.recognize(self, image: Image.Image) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `io.BytesIO`, `image.save`, `subprocess.run`, `str`, `stream.getvalue`, `process.stdout.decode`, `text.strip`
- 반환 표현: `{'text': text, 'status': 'COMPLETE' if text.strip() else 'NO_TEXT'}`

**`_torch_setup` — L91**
```python
_torch_setup(device: str) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.cuda.is_available`, `ValueError`, `torch.set_num_threads`, `torch.set_float32_matmul_precision`
- 반환 표현: `torch`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_unit_vectors` — L103**
```python
_unit_vectors(tensor, dimension: int) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tensor.detach().to`, `tensor.detach`, `torch.isfinite(value).all`, `torch.isfinite`, `ValueError`, `value.norm`, `(norms <= 0).any`
- 반환 표현: `value / norms`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `SSCD` — L115**

기저: `object`.

**`SSCD.__init__` — L116**
```python
SSCD.__init__(self, artifact: Path, device: str='cpu') -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_torch_setup`, `Path(artifact).resolve`, `Path`, `audit.sha`, `ValueError`, `_identity`, `self.torch.jit.load(str(artifact), map_location=device).float().eval`, `self.torch.jit.load(str(artifact), map_location=device).float`, `self.torch.jit.load`, `str` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`SSCD.image_vectors` — L135**
```python
SSCD.image_vectors(self, images: list[Image.Image]) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.torch.stack([self.transform(image) for image in images]).to`, `self.torch.stack`, `self.transform`, `self.torch.inference_mode`, `_unit_vectors`, `self.model`, `vector.tolist`
- 반환 표현: `[{'vector': vector.tolist()} for vector in vectors]`

**`text_chunks` — L142**
```python
text_chunks(tokenizer, text: str) -> tuple[list[list[int]], list[int]]
```
- 기존 설명: Keep every encoded token; SigLIP EOS doubles as the final padding token.
- 직접 호출: `tokenizer.encode`, `type`, `ValueError`, `range`, `len`, `lengths.append`, `chunks.append`
- 반환 표현: `([[eos] + [pad] * 63], [0])`; `(chunks, lengths)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `SigLIP` — L159**

기저: `object`.

**`SigLIP.__init__` — L160**
```python
SigLIP.__init__(self, snapshot: Path, device: str='cpu') -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_torch_setup`, `Path(snapshot).absolute`, `Path`, `ValueError`, `_identity`, `AutoProcessor.from_pretrained`, `AutoModel.from_pretrained(snapshot, local_files_only=True, torch_dtype=self.torch.float32, …`, `AutoModel.from_pretrained(snapshot, local_files_only=True, torch_dtype=self.torch.float32, …`, `AutoModel.from_pretrained(snapshot, local_files_only=True, torch_dtype=self.torch.float32, …`, `AutoModel.from_pretrained`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`SigLIP.image_vectors` — L184**
```python
SigLIP.image_vectors(self, images: list[Image.Image]) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.processor.image_processor(images=images, return_tensors='pt')['pixel_values'].to`, `self.processor.image_processor`, `tuple`, `ValueError`, `self.torch.inference_mode`, `_unit_vectors`, `self.model.get_image_features`, `vector.tolist`
- 반환 표현: `[{'vector': vector.tolist()} for vector in vectors]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`SigLIP.text_vectors` — L192**
```python
SigLIP.text_vectors(self, texts: list[str]) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `text_chunks`, `range`, `len`, `self.torch.tensor`, `self.torch.inference_mode`, `vectors.append`, `_unit_vectors`, `self.model.get_text_features`, `self.torch.cat`, `vectors[offset:offset + len(chunks)].mean` 외 4개
- 반환 표현: `[]`; `result`

**`load_manifest` — L214**
```python
load_manifest(path: Path, *, texts: bool=False) -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `vault._read`, `Path`, `type`, `any`, `isinstance`, `_SHA.fullmatch`, `values.items`, `ValueError`, `c.normalize_text` 외 4개
- 반환 표현: `values`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_entry` — L229**
```python
_entry(path: Path, engine_sha: str, source_sha: str, kind: str) -> dict | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.exists`, `path.is_symlink`, `json.loads`, `vault._read`, `value.get`, `c._digest`, `ValueError`
- 반환 표현: `None`; `value['payload']`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`produce` — L239**
```python
produce(engine, sources: dict[str, str], output_root: Path, *, texts: bool=False, max_items: int | None=None, batch_size: int=8, workers: int=2) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `isinstance`, `_SHA.fullmatch`, `sources.items`, `type`, `ValueError`, `c._digest`, `_private_root`, `Path`, `vault._safe_root` 외 21개
- 반환 표현: `{'status': 'PARTIAL_CACHE_ONLY', 'cache_root': str(root), 'engine_sha256': engine_sha, 'requested_entries': len(sources), 'selected_entries': len(keys), 'new_entries': created, 'reused_ …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L327**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_mutually_exclusive_group`, `source.add_argument`, `parser.parse_args`, `bool`, `parser.error`, `vault._safe_root`, `str`, `root.iterdir` 외 9개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `SystemExit`; 전파되는 callee 예외는 별도.

### D18.060 `fine_tuning/notebook_rl/export.py`

[현재 소스](../../fine_tuning/notebook_rl/export.py) · SHA256 `2e0b66f7895ea326d8c2b242f45d0e1371b5b702ac813a1e42d920f8c147d14c`

모듈의 기존 설명: Out-of-place FP32 fusion → sharded BF16 HF export, with immutable sources.

**`validate_export_identity` — L25**
```python
validate_export_identity(model_path: Path, arm: str) -> dict
```
- 기존 설명: Validate a completed derived export without the historical evaluator package.
- 직접 호출: `_safe_root`, `Path`, `json.loads`, `_read`, `isinstance`, `manifest.get`, `ValueError`, `files.items`, `len`, `PureWindowsPath(name).is_absolute` 외 9개
- 반환 표현: `{'arm': arm, 'manifest_sha256': digest(manifest), 'files': files, 'derived_precision_policy': manifest['precision_policy']}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_base` — L56**
```python
_validate_base(model_path: Path, manifest_path: Path) -> tuple[dict, dict[str, str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `manifest_path.read_text`, `model_path.is_dir`, `ValueError`, `Path(name).is_absolute`, `Path`, `len`, `(model_path / name).stat`, `file_hash`, `files.keys` 외 3개
- 반환 표현: `(manifest, files)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`export_merged` — L76**
```python
export_merged(ckpt: Path, model_path: Path, out: Path, arm: str, *, binding: dict, max_shard_size: int=2000000000, _test_policy_factory: PolicyFactory | None=None, _test_base_manifest: Path | None=None) -> dict
```
- 기존 설명: Streaming shards contain canonical unique parameters, so tied weights are not duplicated.
- 직접 호출: `Path`, `Path(__file__).resolve`, `out.resolve().is_relative_to`, `out.resolve`, `model_path.resolve`, `ckpt.resolve`, `ValueError`, `out.exists`, `out.is_symlink`, `FileExistsError` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`; 전파되는 callee 예외는 별도.

**`_numel` — L174**
```python
_numel(shape: list[int]) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `result`

### D18.061 `fine_tuning/notebook_rl/fact_context_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/fact_context_probe.py) · SHA256 `6b6147c5161360b25e3964fd126f3be588b34dd710e5418e45a455c1fba853bb`

모듈의 기존 설명: Paired local diagnosis; no training or accuracy claim.

**`contract` — L60**
```python
contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`
- 반환 표현: `{'protocol_id': VERSION, 'runtime_protocol_sha256': protocol['sha256'], 'prompt': FACTS, 'conditions': list(CONDITIONS), 'n': 4, 'max_new_tokens': 1024, 'context_limit': 16384, 'max_sec …`

**`output_path` — L72**
```python
output_path(output: Path, *protected: Path) -> Path
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.resolve`, `output.exists`, `any`, `resolved.is_relative_to`, `p.resolve`, `p.resolve().is_relative_to`, `ValueError`, `artifact_path`
- 반환 표현: `artifact_path(output)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_page_journal` — L82**
```python
_page_journal(trees, page_events, diag_positions, tree_positions) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tree.notebook.get`, `note.get`, `type`, `ValueError`, `pages.plan`, `episode_notebook.solution_samples`, `len`, `any`, `enumerate`, `zip` 외 7개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`closed_parent` — L119**
```python
closed_parent(parent: Path, actors: list[ActorItem], protocol: dict, model_path: Path, *, include_notebooks: bool=False, pair_classification: bool=False, whole_synthesis: bool=False) -> tuple[list[ProblemTree], dict]
```
- 기존 설명: Accept a closed minimal run, including a genuine zero-learning failure.
- 직접 호출: `protocol.get`, `read`, `build_protocol`, `checkpoint.digest`, `any`, `ValueError`, `len`, `set`, `type`, `launch.get` 외 38개
- 반환 표현: `(trees, {'files': {str(p): checkpoint.file_hash(p) for p in paths}, 'source_binding': binding, 'status': 'CLOSED'})`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`relationship_protocol_bridge` — L255**
```python
relationship_protocol_bridge(historical: dict, current: dict) -> dict
```
- 기존 설명: A named V35 bridge; historical approvals never become current approvals.
- 직접 호출: `restore_protocol`, `json.loads`, `json.dumps`, `expected.pop`, `checkpoint.digest`, `historical.get`, `ValueError`
- 반환 표현: `{'version': 'closed-v35-relationship-probe-parent-v1', 'historical_protocol_sha256': historical['sha256'], 'current_protocol_sha256': current['sha256'], 'added_policy': policy, 'histori …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`relationship_parent` — L272**
```python
relationship_parent(parent: Path, actors: list[ActorItem], protocol: dict, model_path: Path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read`, `relationship_protocol_bridge`, `closed_parent`, `checkpoint.file_hash`, `ValueError`, `inventory.items`, `Path(name).is_absolute`, `Path`, `path.resolve().is_relative_to`, `path.resolve` 외 17개
- 반환 표현: `(trees, evidence)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`pair_classification_parent` — L322**
```python
pair_classification_parent(parent: Path, actors: list[ActorItem], protocol: dict, model_path: Path) -> unannotated
```
- 기존 설명: Bind the named zero-update V41 history without relabelling its old data approval.
- 직접 호출: `_historical_notebook_parent`
- 반환 표현: `_historical_notebook_parent(parent, actors, protocol, model_path, whole_synthesis=False)`

**`whole_synthesis_parent` — L327**
```python
whole_synthesis_parent(parent: Path, actors: list[ActorItem], protocol: dict, model_path: Path) -> unannotated
```
- 기존 설명: Bind only the audited zero-update V45 capture; current approvals remain separate.
- 직접 호출: `parent.resolve`, `(WHOLE_SYNTHESIS_ROOT / 'minimal').resolve`, `checkpoint.file_hash`, `ValueError`, `_historical_notebook_parent`
- 반환 표현: `_historical_notebook_parent(parent, actors, protocol, model_path, whole_synthesis=True)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_historical_notebook_parent` — L336**
```python
_historical_notebook_parent(parent: Path, actors: list[ActorItem], protocol: dict, model_path: Path, *, whole_synthesis: bool) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `closed_parent`, `read`, `checkpoint.file_hash`, `ValueError`, `audit.get`, `str`, `audit['read_metadata_sha256'].items`, `Path`, `path.is_symlink`, `paths.append` 외 21개
- 반환 표현: `(trees, evidence)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_requests` — L423**
```python
paired_requests(actors, trees, images, prompts) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `ValueError`, `enumerate`, `zip`, `public_transcript`, `solution_samples`, `for_actor`, `seeds`, `wire(public if condition == 'WITH_ALL_HISTORY' else []).decode`, `wire` 외 9개
- 반환 표현: `requests`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`bind_history_tokens` — L452**
```python
bind_history_tokens(trees, engine, probes) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `zip`, `engine.prepare`, `solution_samples`, `engine.tokenizer.decode`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`trained_parent` — L464**
```python
trained_parent(parent: Path, reload: Path, model_path: Path, cache_check: Path) -> dict
```
- 기존 설명: A real saved update plus a closed fresh reload, without authorizing fixed-ten training.
- 직접 호출: `parent_gate`, `read`, `type`, `restored.get`, `psutil.pid_exists`, `Path(launch['model_path']).resolve`, `Path`, `model_path.resolve`, `Path(launch['cache_check']).resolve`, `cache_check.resolve` 외 1개
- 반환 표현: `gate`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`trained_note_counts` — L480**
```python
trained_note_counts(rows: list[dict]) -> dict
```
- 기존 설명: Strict paired notes on already-seen history, not new-question accuracy.
- 직접 호출: `any`, `type`, `ValueError`, `len`, `sum`
- 반환 표현: `{'notes': len(pairs), 'before_strict_valid': sum((a == 1 for a, _ in pairs)), 'after_strict_valid': sum((b == 1 for _, b in pairs)), 'recovered': sum((a == 0 and b == 1 for a, b in pair …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`collect` — L494**
```python
collect(requests, engine, verifier, record, deadline: float) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `time.monotonic`, `TimeoutError`, `engine.prepare`, `len`, `ValueError`, `tensor_identity`, `prepared.pixels.items`, `list`, `image_bindings.setdefault`, `record` 외 8개
- 반환 표현: `rows`
- 명시적 예외: `TimeoutError`, `ValueError`; 전파되는 callee 예외는 별도.

**`capture` — L543**
```python
capture(preparation: Path, model_path: Path, cache_check: Path, parent: Path, output: Path, *, notebook_layout: bool=False, notebook_drafts: bool=False, trained_notebook_reload: Path | None=None, notebook_relationships: bool=False, notebook_relationship_comparisons: bool=False, notebook_pair_classification: bool=False, notebook_whole_synthesis: bool=False, notebook_whole_reference: bool=False, pair_to_whole: str | None=None, pair_to_whole_baseline: Path | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sum`, `bool`, `ValueError`, `int`, `os.environ.get`, `os.getppid`, `time.monotonic`, `enable_deterministic_execution`, `trained_parent`, `restore_protocol` 외 80개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`main` — L803**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_mutually_exclusive_group`, `diagnostic.add_argument`, `parser.parse_args`, `any`, `parser.error`, `signal.signal`, `capture`, `str` 외 9개
- 반환 표현: `0`; `process.wait(timeout=MAX_SECONDS - 30)`; `124`

### D18.062 `fine_tuning/notebook_rl/fact_credit.py`

[현재 소스](../../fine_tuning/notebook_rl/fact_credit.py) · SHA256 `18c8b710429d6efa9fc492fe2f6bd35494ac5f5c33a9e2785ae3b59e4344fcd5`

모듈의 기존 설명: Host-only credit for generated JSON facts; no correct values or teacher targets.

**`_bad_constant` — L12**
```python
_bad_constant(value) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`value_spans` — L16**
```python
value_spans(text: str) -> dict[str, tuple[int, int]]
```
- 기존 설명: Map canonical IDs to original JSON value literals, rejecting alias collisions.
- 직접 호출: `json.JSONDecoder`, `decoder.decode`, `type`, `parsed.get`, `ValueError`, `next`, `members`, `slot_id`
- 반환 표현: `spans`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`token_spans` — L60**
```python
token_spans(tokenizer, ids: list[int], text: str) -> list[tuple[int, int]] | None
```
- 기존 설명: Abstain on unstable byte-token prefixes; full text disagreement is an error.
- 직접 호출: `tokenizer.decode`, `ValueError`, `range`, `len`, `prefix.startswith`, `text.startswith`, `spans.append`
- 반환 표현: `None`; `spans`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`group_credit` — L75**
```python
group_credit(samples, scores: dict[str, dict], tokenizer) -> dict[str, dict]
```
- 기존 설명: Full-group means precede MIS. Missing/invalid notes contribute zeros, not labels.
- 직접 호출: `len`, `ValueError`, `row.get`, `any`, `type`, `counts.pop`, `isinstance`, `zip`, `digest`, `slot_id` 외 9개
- 반환 표현: `output`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.063 `fine_tuning/notebook_rl/fresh_observation.py`

[현재 소스](../../fine_tuning/notebook_rl/fresh_observation.py) · SHA256 `ba991cc9fb77772719c3370c58fce50f95830f06b10b0563bd23be20c2412ee5`

모듈의 기존 설명: One unverified, same-policy image reread; context only, never a rewarded action.

**`request_for` — L29**
```python
request_for(actor: ActorItem, schema: str, images: list, policy_version: int, master_seed: int) -> RolloutRequest
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `isinstance`, `schema.strip`, `len`, `ValueError`, `p0_body(actor.question, actor.choices).removesuffix`, `p0_body`, `RolloutRequest`, `user_message`, `PROMPT.format` 외 1개
- 반환 표현: `RolloutRequest(f'{actor.problem_id}:v{policy_version}:FRESH_OBSERVATION', None, [user_message(PROMPT.format(question=question, schema=schema), len(images))], images, 1, MAX_NEW_TOKENS,  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate` — L42**
```python
generate(actor: ActorItem, schema: str, images: list, engine: Any, *, master_seed: int, context_limit: int, on_event: Callable[[dict], None] | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `request_for`, `engine.prepare`, `len`, `ValueError`, `json.loads`, `to_json`, `list`, `on_event`, `time.monotonic`, `_sample_request` 외 5개
- 반환 표현: `receipt`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate` — L74**
```python
validate(receipt: dict, *, policy_version: int, tokenizer: Any | None=None, preparation: Any | None=None, images: list | None=None, actor: ActorItem | None=None, schema: str | None=None, master_seed: int | None=None) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `from_json`, `json.dumps`, `ValueError`, `request_for`, `len`, `receipt.get`, `_hash`, `receipt.items`, `any`, `type` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`quoted_context` — L114**
```python
quoted_context(receipt: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps`, `value.replace('<', '\\u003c').replace`, `value.replace`
- 반환 표현: `'An independent reread by this same policy, without prior attempts, follows. It is model-generated and UNVERIFIED, not a correct oracle or an additional solution attempt. Compare it wit …`

### D18.064 `fine_tuning/notebook_rl/gate_cache.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_cache.py) · SHA256 `7276301fd90e05dfb8941512306c0dbe4a9066f3ca11595a147635af4e2efb99`

모듈의 기존 설명: G6 cache measurements across caller-owned, sequential native processes.

**`_require` — L26**
```python
_require(value: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_process` — L31**
```python
_process() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `psutil.Process`, `process.ppid`, `process.create_time`, `Path('/proc/sys/kernel/random/boot_id').read_text().strip`, `Path('/proc/sys/kernel/random/boot_id').read_text`, `Path`
- 반환 표현: `{'pid': process.pid, 'parent_pid': process.ppid(), 'create_time': process.create_time(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}`

**`_closed_previous` — L38**
```python
_closed_previous(report: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `report.get`, `_require`, `set`, `type`, `os.getpid`, `math.isfinite`, `_process`, `psutil.Process`, `process.create_time`, `process.is_running`
- 반환 표현: `previous`

**`_dedicated_process` — L56**
```python
_dedicated_process() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `torch.cuda.is_available`, `torch.cuda.is_bf16_supported`, `torch.cuda.memory_allocated`, `psutil.Process().parents`, `psutil.Process`, `subprocess.check_output`, `ancestors.intersection`, `int`, `s.strip` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_probes` — L67**
```python
_probes(probes: list[SyncProbe]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `len`, `all`, `type`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_state` — L75**
```python
_state(llm, config) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_layout`, `_weight_inventory`, `llm.apply_model`, `_read_worker`, `_require`, `isinstance`, `len`
- 반환 표현: `{'parameters': {row['worker_name']: {'shape': row['shape'], 'dtype': row['dtype'], 'sha256': row['sha256']} for row in result['parameters']}, 'device': result['device']}`

**`_input` — L86**
```python
_input(preparation, probe: SyncProbe) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `preparation.prepare`, `list`, `_pixels`
- 반환 표현: `{'request_id': probe.request.request_id, 'prompt_token_ids': list(prepared.token_ids), 'completion_ids': list(probe.completion_ids), 'seed': probe.request.seeds[0], 'image': _pixels(pre …`

**`_probabilities` — L93**
```python
_probabilities(values: list[float], count: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `len`, `all`, `type`, `math.isfinite`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_capture` — L98**
```python
_capture(engine, probes: list[SyncProbe], stage: str, record: Callable[[dict], None], prefix_actions: list[list[int]] | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_probes`, `_state`, `enumerate`, `_input`, `engine.probe_logprobs`, `record`, `_probabilities`, `len`, `_require`, `type` 외 3개
- 반환 표현: `{'stage': stage, 'pid': os.getpid(), 'process': _process(), 'state': state, 'probes': rows, 'training_allowed': False, 'gpu_gates_certified': False}`

**`measure_current` — L124**
```python
measure_current(engine, probes: list[SyncProbe], *, stage: str, record: Callable[[dict], None]) -> dict
```
- 기존 설명: Capture the same actual engine BEFORE and AFTER the caller's real update/push.
- 직접 호출: `_require`, `callable`, `require_idle_engine`, `_pinned_config`, `_capture`, `result.update`
- 반환 표현: `result`

**`measure_fresh` — L139**
```python
measure_fresh(model_path: Path, image_cfg: dict, *, kv_cache_memory_bytes: int, max_num_seqs: int, max_model_len: int, probes: list[SyncProbe], control: dict, record: Callable[[dict], None]) -> dict
```
- 기존 설명: Allocate only after the old process exits; always close this isolated engine.
- 직접 호출: `_require`, `callable`, `control.get`, `os.getpid`, `len`, `_closed_previous`, `_probes`, `_dedicated_process`, `VLLMRolloutEngine`, `require_idle_engine` 외 7개
- 반환 표현: `result`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`_cache_trace` — L170**
```python
_cache_trace(manager, runner, model, stack: ExitStack, hits: list[dict], visual: list[dict]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `vars`, `vars(manager).get`, `stack.callback`, `_observe_visual`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_control_call` — L192**
```python
_control_call(llm, preparation, probe, *, prefix: bool, hits, visual, record) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `preparation.prepare`, `_require`, `len`, `VLLMRolloutEngine._prompt`, `range`, `import_module('vllm').SamplingParams`, `import_module`, `llm.generate`, `list`, `str` 외 5개
- 반환 표현: `{'token_ids': action, 'logprobs': probabilities, 'num_cached_tokens': output.num_cached_tokens, 'encoder_checks': hits[start:], 'visual_call_count': len(visual) - visuals}`

**`_uncleared_load` — L237**
```python
_uncleared_load(llm, named: Iterable[tuple[str, torch.Tensor]], shapes: dict) -> None
```
- 기존 설명: Deliberate negative control; never mark a training engine synchronized.
- 직접 호출: `set`, `_require`, `llm.apply_model`, `shapes.keys`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`measure_default_control` — L253**
```python
measure_default_control(model_path: Path, image_cfg: dict, *, kv_cache_memory_bytes: int, max_num_seqs: int, max_model_len: int, post_weights: Iterable[tuple[str, torch.Tensor]], previous: dict, probes: list[SyncProbe], record: Callable[[dict], None]) -> dict
```
- 기존 설명: Allocate one disposable native LLM in its OWN process, never beside HF.
- 직접 호출: `_probes`, `_require`, `previous.get`, `_closed_previous`, `callable`, `version`, `_dedicated_process`, `AutoConfig.from_pretrained`, `str`, `_pinned_config` 외 24개
- 반환 표현: `{'stage': 'DEFAULT_CONTROL', 'pid': os.getpid(), 'process': _process(), 'closed_predecessor': predecessor, 'state_before': before, 'state_after': after, 'probes': rows, 'cache_settings' …`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`_difference` — L329**
```python
_difference(a: list[float], b: list[float]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_probabilities`, `len`, `abs`, `zip`, `sum`, `max`
- 반환 표현: `{'mean_abs': sum(values) / len(values), 'max_abs': max(values)}`

**`_compare` — L336**
```python
_compare(before: dict, after: dict, control: dict, fresh: dict, threshold: float) -> dict
```
- 기존 설명: Arithmetic seam for authored CPU tests; no native execution certification.
- 직접 호출: `_require`, `type`, `math.isfinite`, `v.get`, `range`, `any`, `k.startswith`, `set`, `len`, `all` 외 3개
- 반환 표현: `{'fresh_engine_match': True, 'default_cache_comparison_measured': True, 'default_mismatch_observed': any((r[k]['max_abs'] > threshold for r in rows for k in ('default_encoder', 'default …`

**`compare_cache` — L383**
```python
compare_cache(before: dict, after: dict, control: dict, fresh: dict, *, threshold: float, current_process, control_process, fresh_process) -> dict
```
- 기존 설명: Compare only completed sequential children; never accept a success boolean.
- 직접 호출: `_require`, `torch.cuda.is_initialized`, `all`, `isinstance`, `p.poll`, `len`, `os.getpid`, `v.get`, `_compare`, `result.update`
- 반환 표현: `result`

### D18.065 `fine_tuning/notebook_rl/gate_checkpoint.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_checkpoint.py) · SHA256 `f1b116456e8d92fac22ead6944b04362b50ceba13ee58433c93d084f1752073f`

모듈의 기존 설명: G12 capture/replay stages for sequential, coordinator-owned processes.

**`_require` — L33**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_process` — L38**
```python
_process() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `psutil.Process`, `process.ppid`, `process.create_time`, `Path('/proc/sys/kernel/random/boot_id').read_text().strip`, `Path('/proc/sys/kernel/random/boot_id').read_text`, `Path`
- 반환 표현: `{'pid': process.pid, 'parent_pid': process.ppid(), 'birth': process.create_time(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}`

**`_exited` — L44**
```python
_exited(identity: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `_process`, `psutil.Process(identity['pid']).create_time`, `psutil.Process`
- 반환 표현: `psutil.Process(identity['pid']).create_time() != identity['birth']`; `True`

**`_state` — L52**
```python
_state(value: Any) -> Any
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `_require`, `bool`, `torch.isfinite(value).all`, `torch.isfinite`, `list`, `str`, `tensor_sha256`, `value.reshape`, `_state` 외 1개
- 반환 표현: `{'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': tensor_sha256(value.reshape(-1))}`; `{str(k): _state(v) for k, v in value.items()}`; `[_state(v) for v in value]`; 외 1개 분기

**`_runtime` — L63**
```python
_runtime(bundle, optimizer, scheduler) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `policy.assert_base_unchanged`, `_state`, `bundle.model.named_parameters`, `optimizer.state_dict`, `scheduler.state_dict`, `checkpoint.capture_rng`
- 반환 표현: `{'parameters': {n: _state(p) for n, p in bundle.model.named_parameters()}, 'optimizer': _state(optimizer.state_dict()), 'scheduler': _state(scheduler.state_dict()), 'rng': _state(checkp …`

**`_sources` — L72**
```python
_sources() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `checkpoint.file_hash`
- 반환 표현: `{name: checkpoint.file_hash(ROOT / 'fine_tuning' / name) for name in names}`

**`_model_files` — L79**
```python
_model_files(model_path: Path, protocol: dict, test_manifest: Path | None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `manifest_path.read_text`, `_require`, `bool`, `len`, `set`, `all`, `Path`, `path.stat`, `checkpoint.file_hash`
- 반환 표현: `{'manifest_sha256': checkpoint.file_hash(manifest_path), 'files': manifest['files']}`

**`_native` — L94**
```python
_native(bundle, preparation, protocol: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `bundle.model.parameters`, `len`, `next`, `iter`, `_pinned_config`, `measure_inventory`, `psutil.Process().parents`, `psutil.Process` 외 6개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**클래스 `_ObservedPreparation` — L116**

기저: `object`.

**`_ObservedPreparation.__init__` — L117**
```python
_ObservedPreparation.__init__(self, preparation) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_ObservedPreparation.model_inputs` — L122**
```python
_ObservedPreparation.model_inputs(self, request, completion, device) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.preparation.model_inputs`, `_require`, `inputs['input_ids'].tolist`, `self.calls.append`, `_input_fingerprint`, `len`
- 반환 표현: `inputs`

**`_batch` — L131**
```python
_batch(trees, images, algorithm: str, seed: int, vision_disabled: bool) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `to_json`, `list`, `hashlib.sha256(im.tobytes()).hexdigest`, `hashlib.sha256`, `im.tobytes`, `images.items`
- 반환 표현: `{'trees': [to_json(t) for t in trees], 'images': {pid: [{'mode': im.mode, 'size': list(im.size), 'pixels_sha256': hashlib.sha256(im.tobytes()).hexdigest()} for im in values] for pid, va …`

**`_observe_and_step` — L139**
```python
_observe_and_step(bundle, optimizer, scheduler, trees, images, preparation, probes, protocol: dict, algorithm: str, step: int, seed: int, vision_disabled: bool, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `all`, `bundle.model.parameters`, `len`, `sorted`, `checkpoint._validate_optimizer`, `_runtime`, `next`, `checkpoint.capture_rng`, `enumerate` 외 14개
- 반환 표현: `{'before': before, 'after': after, 'batch': batch, 'probes': measured, 'input_calls': observed.calls, 'update': {k: v for k, v in update.items() if k != 'seconds'}, 'logits_sha256': che …`

**`_stage` — L189**
```python
_stage(output: Path, operation) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.mkdir`, `_write`, `_process`, `operation`, `type`, `str`, `traceback.format_exc`
- 반환 표현: `record`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`_capture_checkpoint` — L202**
```python
_capture_checkpoint(bundle, optimizer, scheduler, trees, images, preparation, probes, protocol: dict, algorithm: str, *, step: int, seed: int, cursor: dict, binding: dict, model_path: Path, output: Path, vision_disabled: bool=False, _test_manifest: Path | None=None) -> dict
```
- 기존 설명: Private CPU seam; supplying a tiny source manifest never yields native evidence.
- 직접 호출: `_stage`
- 반환 표현: `_stage(output, operation)`

**`capture_checkpoint` — L225**
```python
capture_checkpoint(bundle, optimizer, scheduler, trees, images, preparation, probes, protocol: dict, algorithm: str, *, step: int, seed: int, cursor: dict, binding: dict, model_path: Path, output: Path, vision_disabled: bool=False) -> dict
```
- 기존 설명: Save and measure once in a dedicated pinned-CUDA child, which must then exit.
- 직접 호출: `_capture_checkpoint`
- 반환 표현: `_capture_checkpoint(bundle, optimizer, scheduler, trees, images, preparation, probes, protocol, algorithm, step=step, seed=seed, cursor=cursor, binding=binding, model_path=model_path, o …`

**`_replay_checkpoint` — L234**
```python
_replay_checkpoint(model_path: Path, capture: Path, output: Path, trees, images, probes, preparation_factory, *, protocol: dict, binding: dict, device: str, _test_manifest: Path | None=None, _test_policy_factory=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_stage`
- 반환 표현: `_stage(output, operation)`

**`replay_checkpoint` — L273**
```python
replay_checkpoint(model_path: Path, capture: Path, output: Path, trees, images, probes, preparation_factory, *, protocol: dict, binding: dict, device: str) -> dict
```
- 기존 설명: Load fresh F0 plus saved state; construct InputPreparation from the new bundle.
- 직접 호출: `_replay_checkpoint`
- 반환 표현: `_replay_checkpoint(model_path, capture, output, trees, images, probes, preparation_factory, protocol=protocol, binding=binding, device=device)`

**`compare_checkpoint` — L280**
```python
compare_checkpoint(capture: Path, replay: Path, capture_process: subprocess.Popen, replay_process: subprocess.Popen, output: Path) -> dict
```
- 기존 설명: Require actual owned exited children, then compare every measured value exactly.
- 직접 호출: `_stage`
- 반환 표현: `_stage(output, operation)`

### D18.066 `fine_tuning/notebook_rl/gate_gradients.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_gradients.py) · SHA256 `e5afc25db7766486e0ace1c51b895f760b3e7bd44058b3ffb581c1358a7d91a5`

모듈의 기존 설명: G9 gradient-route observations on a caller-owned policy; no optimizer update.

**`_require` — L18**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_fingerprints` — L23**
```python
_fingerprints(inputs: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `str`, `tensor_sha256`, `inputs.items`
- 반환 표현: `{name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device), 'sha256': tensor_sha256(value)} for name, value in inputs.items()}`

**`_capture` — L28**
```python
_capture(bundle, names: dict[int, str], retain: set[str]) -> tuple[dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `id`, `bundle.params.values`, `_require`, `all`, `bundle.model.parameters`, `bundle.params.items`, `rows.append`, `bool`, `torch.isfinite(grad).all`, `torch.isfinite` 외 10개
- 반환 표현: `(summary, gradients)`

**`_match` — L55**
```python
_match(expected: dict, actual: dict, *, comparison: str='GRADIENT') -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `set`, `bool`, `sorted`, `list`, `str`, `json.dumps`, `(observed.double() - reference.double()).abs`, `observed.double`, `reference.double` 외 11개
- 반환 표현: `{'match': True, 'rtol': 1e-06, 'atol': 0.0, 'parameters': rows, 'max_abs_error': max((row['max_abs_error'] for row in rows)), 'max_relative_error': max((row['max_relative_error'] for ro …`

**`_measure_routes` — L79**
```python
_measure_routes(bundle, sample_inputs: dict, completion_len: int, *, native_cuda: bool=False, chunk: int=1024) -> dict
```
- 기존 설명: Private tiny-CPU seam; native_cuda is reserved for the strict public path.
- 직접 호출: `dict`, `bundle.model.named_parameters`, `list`, `named.values`, `_require`, `bool`, `all`, `torch.is_inference_mode_enabled`, `set`, `isinstance` 외 30개
- 반환 표현: `report`

**`measure_gradient_routes` — L247**
```python
measure_gradient_routes(bundle, sample_inputs: dict, completion_len: int, *, protocol: dict) -> dict
```
- 기존 설명: G9 component for a live pinned CUDA policy; allocates no model or optimizer.
- 직접 호출: `_require`, `type`, `all`, `bundle.model.parameters`, `len`, `next`, `iter`, `_pinned_config`, `measure_inventory`, `_measure_routes` 외 1개
- 반환 표현: `result`

### D18.067 `fine_tuning/notebook_rl/gate_guards.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_guards.py) · SHA256 `dcea80d752d4ebf393ddeb0699df0e385929f38e87f68c45d765f9c8fcd6cdd5`

모듈의 기존 설명: G13 observations around ONE existing G4/G11 production reward update.

**`_require` — L33**
```python
_require(value: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_native` — L38**
```python
_native(bundle, preparation, protocol: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `bundle.model.parameters`, `len`, `next`, `iter`, `_pinned_config`, `measure_inventory`, `load_protocol`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_eligibility` — L55**
```python
_eligibility(root: Path, request: dict, protocol: dict, validator=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `set`, `type`, `Path(value['path']).is_absolute`, `Path`, `validator or approval.validate_for_gate_data`, `result.get`, `','.join`, `from_json`, `json.dumps` 외 3개
- 반환 표현: `result | {'actors': actors}`

**`_image` — L75**
```python
_image(image) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `hashlib.sha256(image.tobytes()).hexdigest`, `hashlib.sha256`, `image.tobytes`
- 반환 표현: `{'size': list(image.size), 'mode': image.mode, 'pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest()}`

**`_bind_batch` — L80**
```python
_bind_batch(trees, images, eligible: dict, root: Path, preparation, protocol: dict, algorithm: str, native: bool) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `len`, `set`, `actors.keys`, `load_prompts`, `load_images`, `all`, `a.getextrema`, `zip`, `_image` 외 10개
- 반환 표현: `{'rows': bound, 'input_binding': eligible['input_binding'], 'native_prompt_checked': native}`

**`_denied` — L117**
```python
_denied(action, name: str, record) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `action`, `type`, `str`, `record`, `hasattr`, `value.close`, `ValueError`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_actor_guards` — L130**
```python
_actor_guards(root: Path, request: dict, forbidden_roots: list[Path], record) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `bool`, `all`, `type`, `isinstance`, `p.is_absolute`, `Path`, `guards.assert_no_teacher_imports`, `guards.install_guards`, `_denied` 외 9개
- 반환 표현: `results`

**`_measure_guards` — L157**
```python
_measure_guards(bundle, optimizer, scheduler, preparation, protocol: dict, *, build_root: Path, data_request: dict, forbidden_roots: list[Path], algorithm: str, step: int, seed: int, record, _test_data_validator=None) -> unannotated
```
- 기존 설명: Private authored-data/tiny-CPU seam; never certifies full data or CUDA gates.
- 직접 호출: `_require`, `callable`, `_native`, `checkpoint._validate_optimizer`, `all`, `bundle.model.parameters`, `Path(build_root).absolute`, `Path`, `_eligibility`, `Image.init` 외 34개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`measure_guards` — L286**
```python
measure_guards(bundle, optimizer, scheduler, preparation, protocol: dict, *, build_root: Path, data_request: dict, forbidden_roots: list[Path], algorithm: str, step: int, seed: int, record) -> unannotated
```
- 기존 설명: Wrap the caller's shared G4/G11 call; no model load or extra optimizer step.
- 직접 호출: `_measure_guards`
- 반환 표현: `_measure_guards(bundle, optimizer, scheduler, preparation, protocol, build_root=build_root, data_request=data_request, forbidden_roots=forbidden_roots, algorithm=algorithm, step=step, s …`

### D18.068 `fine_tuning/notebook_rl/gate_images.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_images.py) · SHA256 `876b35ebe49d36b61986d346b7c3c21ff91860a2606d2ab84d1ac30980fda977`

모듈의 기존 설명: G2 observations from the actual worker visual forward, never CPU expectations alone.

**`_image_rows` — L20**
```python
_image_rows(pixels: torch.Tensor, grid: list[list[int]], merge: int) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `bool`, `torch.isfinite(pixels).all`, `torch.isfinite`, `type`, `any`, `len`, `ValueError`, `rows.append`, `list` 외 1개
- 반환 표현: `rows`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_observe_visual` — L42**
```python
_observe_visual(model, stack: ExitStack, records: list[dict]) -> dict
```
- 기존 설명: Private seam tests hook semantics with authored modules, not a native GPU.
- 직접 호출: `len`, `ValueError`, `visual.register_forward_hook`, `stack.callback`
- 반환 표현: `{'merge_size': merge, 'feature_width': width}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_compare` — L68**
```python
_compare(prepared: Prepared, observed: list[dict], image_token_id: int) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_image_rows`, `pixels['pixel_values'].to`, `pixels['image_grid_thw'].tolist`, `RuntimeError`, `prepared.token_ids.count`, `sum`, `hashlib.sha256(json.dumps(prepared.token_ids, separators=(',', ':')).encode()).hexdigest`, `hashlib.sha256`, `json.dumps(prepared.token_ids, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `{'expected_images': expected, 'observed_calls': observed, 'encoder_order_matches_input': expected == actual, 'image_tokens': token_count, 'prompt_sha256': hashlib.sha256(json.dumps(prep …`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`measure_image_parity` — L84**
```python
measure_image_parity(engine, requests: list[RolloutRequest]) -> dict
```
- 기존 설명: Observe fixed single-/multi-image probes in an already-owned idle engine.
- 직접 호출: `require_idle_engine`, `len`, `sorted`, `any`, `ValueError`, `ExitStack`, `engine.llm.apply_model`, `RuntimeError`, `replace`, `engine.prepare` 외 4개
- 반환 표현: `{'images_checked': sum((len(p['expected_images']) for p in probes)), 'grid_mismatches': 0, 'token_feature_mismatches': 0, 'pixel_bf16_mismatches': 0, 'runtime': runtime, 'probes': probe …`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

### D18.069 `fine_tuning/notebook_rl/gate_inputs.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_inputs.py) · SHA256 `85aa29f9cc2955a1e735bacfe911520ca2d83c1db4648556ad79fe1f4bd44901`

모듈의 기존 설명: Private, write-once G12 input handoff; not data/GPU/training approval.

**`_require` — L26**
```python
_require(value, message) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_pixels` — L31**
```python
_pixels(images) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `hashlib.sha256(im.tobytes()).hexdigest`, `hashlib.sha256`, `im.tobytes`
- 반환 표현: `[{'mode': im.mode, 'size': list(im.size), 'pixels_sha256': hashlib.sha256(im.tobytes()).hexdigest()} for im in images]`

**`_load_images` — L36**
```python
_load_images(actor, image_root, stack) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `zip`, `load_images`, `replace`, `stack.callback`, `result.append`
- 반환 표현: `result`

**`_record` — L46**
```python
_record(cls, value) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `from_json`, `json.dumps`
- 반환 표현: `from_json(cls, json.dumps(value, allow_nan=False))`

**`_validate` — L50**
```python
_validate(packet, protocol, binding) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `set`, `checkpoint._binding`, `checkpoint._cursor`, `len`, `_record`, `packet['actors'].items`, `bool`, `isinstance` 외 5개
- 반환 표현: `(trees, actors)`

**`save_inputs` — L107**
```python
save_inputs(path: Path, *, trees, images, actors, probes, probe_actors, preparation, image_root: Path, protocol: dict, binding: dict, step: int, algorithm: str, seed: int, cursor: dict) -> dict
```
- 기존 설명: Freeze actual caller-owned next-step inputs without creating/changing images.
- 직접 호출: `_safe_root`, `_require`, `path.is_absolute`, `any`, `p.is_symlink`, `path.resolve().is_relative_to`, `path.resolve`, `image_root.resolve`, `set`, `json.loads` 외 11개
- 반환 표현: `{'path': str(path), 'file_sha256': _sha(raw)}`

**클래스 `ReplayInputs` — L149**

기저: `object`.

```python
trees: list[ProblemTree]
images: dict
probes: list[SyncProbe]
step: int
algorithm: str
seed: int
cursor: dict
```

**`load_inputs` — L159**
```python
load_inputs(descriptor: dict, *, image_root: Path, protocol: dict, binding: dict, stack: ExitStack) -> ReplayInputs
```
- 기존 설명: Reconstruct private inputs; caller's stack owns every opened image.
- 직접 호출: `launch.descriptor`, `_safe_root`, `_require`, `path.is_symlink`, `_read`, `_sha`, `launch.read_json`, `_validate`, `ExitStack`, `_load_images` 외 10개
- 반환 표현: `result`

**`capture_from_inputs` — L190**
```python
capture_from_inputs(bundle, optimizer, scheduler, preparation, descriptor: dict, *, image_root: Path, protocol: dict, binding: dict, model_path: Path, output: Path) -> dict
```
- 기존 설명: Consume the frozen packet for the real G12 capture/update, then bind it.
- 직접 호출: `ExitStack`, `load_inputs`, `_require`, `capture_checkpoint`, `checkpoint.file_hash`, `_write`, `_bytes`
- 반환 표현: `result`

**`replay_from_inputs` — L211**
```python
replay_from_inputs(model_path: Path, capture: Path, output: Path, descriptor: dict, *, image_root: Path, protocol: dict, binding: dict, device: str) -> dict
```
- 기존 설명: Fresh owned child replay, preserving loaded RNG and exact action inputs.
- 직접 호출: `launch.read_json`, `_require`, `checkpoint.file_hash`, `ExitStack`, `load_inputs`, `_batch`, `replay_checkpoint`
- 반환 표현: `replay_checkpoint(model_path, capture, output, inputs.trees, inputs.images, inputs.probes, preparation_factory, protocol=protocol, binding=binding, device=device)`

### D18.070 `fine_tuning/notebook_rl/gate_logprobs.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_logprobs.py) · SHA256 `c892dd7066e5d01110ea47b0fd9b5544a8fedc94fed07ec4c2695e9b814f2faa`

모듈의 기존 설명: G7 measurements from generated sampler actions and the actual HF policy.

**`_require` — L22**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_distribution` — L27**
```python
_distribution(values: list[float]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.tensor`, `_require`, `bool`, `torch.isfinite(tensor).all`, `torch.isfinite`, `len`, `min`, `max`, `float`, `tensor.mean` 외 1개
- 반환 표현: `{'count': len(values), 'min': min(values), 'max': max(values), 'mean': float(tensor.mean()), 'p50': float(torch.quantile(tensor, 0.5)), 'p95': float(torch.quantile(tensor, 0.95)), 'p995 …`

**`_limits` — L35**
```python
_limits(threshold: float, delta_seq: float) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `math.isfinite`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_summarize` — L41**
```python
_summarize(rows: list[dict], threshold: float, delta_seq: float) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_limits`, `_require`, `len`, `torch.tensor(row['hf_logprobs'], dtype=torch.float64).reshape`, `torch.tensor`, `torch.tensor(row['sampler_logprobs'], dtype=torch.float64).reshape`, `old.numel`, `bool`, `torch.isfinite(old).all`, `torch.isfinite` 외 21개
- 반환 표현: `{'trajectories': len(rows), 'tokens': len(absolute), 'mean_abs': sum(absolute) / len(absolute), 'max_abs': max(absolute), 'token_abs_delta': _distribution(absolute), 'sequence_sum_delta …`

**`_measure_logprobs` — L77**
```python
_measure_logprobs(bundle, engine, requests: list[RolloutRequest], *, threshold: float, delta_seq: float, record: Callable[[dict], None]) -> dict
```
- 기존 설명: Private CPU seam: real tiny HF sampling is possible, never native evidence.
- 직접 호출: `_limits`, `_require`, `callable`, `bool`, `len`, `sum`, `list`, `bundle.model.parameters`, `all`, `next` 외 34개
- 반환 표현: `result | {'policy_version': version, 'device': str(device), 'rows': rows, 'native_execution': False, 'training_allowed': False, 'gpu_gates_certified': False, 'thresholds_modified': Fals …`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`measure_logprobs` — L167**
```python
measure_logprobs(bundle, engine, requests: list[RolloutRequest], protocol: dict, *, stage: str, record: Callable[[dict], None]) -> dict
```
- 기존 설명: Measure INITIAL or POST_SYNC from the caller-owned pinned native engine.
- 직접 호출: `_require`, `measure_inventory`, `_pinned_config`, `require_idle_engine`, `next`, `bundle.model.parameters`, `str`, `_measure_logprobs`, `result.update`, `result.pop`
- 반환 표현: `result`

### D18.071 `fine_tuning/notebook_rl/gate_numerics.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_numerics.py) · SHA256 `61f05e9797090bd482004127da1ed7c2f6d3bd468cdece51bfdb5b36ec44e6d0`

모듈의 기존 설명: G10 synthetic numerical/autograd measurements on the explicitly requested device.

**`_scalar` — L30**
```python
_scalar(cur, old, ref, adv, mask, keep, norm: float, beta: float) -> unannotated
```
- 기존 설명: Scalar arithmetic and analytic derivative, away from clipping boundaries.
- 직접 호출: `len`, `enumerate`, `math.exp`, `min`, `max`, `math.expm1`
- 반환 표현: `(total, gradient)`

**`_trl_dr_grpo` — L49**
```python
_trl_dr_grpo(cur, old, ref, adv, mask, cap: int, beta: float) -> unannotated
```
- 기존 설명: Independent transcription of the scoped, pinned TRL expressions above.
- 직접 호출: `cur.float`, `old.detach().float`, `old.detach`, `ref.detach().float`, `ref.detach`, `adv.detach().float().unsqueeze`, `adv.detach().float`, `adv.detach`, `torch.exp`, `torch.clamp` 외 2개
- 반환 표현: `(per_token * mask).sum() / (per_token.shape[0] * cap)`

**`measure_numerics` — L61**
```python
measure_numerics(device: str | torch.device) -> dict
```
- 기존 설명: Return observed checks, or raise; CPU results never imply CUDA execution.
- 직접 호출: `torch.device`, `ValueError`, `torch.cuda.is_available`, `RuntimeError`, `torch.cuda.device`, `torch.cuda.is_bf16_supported`, `torch.empty`, `torch.enable_grad`, `str(dtype).removeprefix`, `str` 외 51개
- 반환 표현: `{'numeric_tests_match': bool(cases) and all((c['match'] for c in cases)), 'trl_formula_match': all((c['match'] for c in cases if c['name'].startswith('trl/'))), 'device': str(target), ' …`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

### D18.072 `fine_tuning/notebook_rl/gate_reference.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_reference.py) · SHA256 `f4e6651d687a062620e2afa5a4988d8b4bfbbfa199bd949f059be014ec7bb1bd`

모듈의 기존 설명: G3 fresh-F0 comparison at the actual completion-prediction positions.

**`_require` — L18**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_input_fingerprint` — L23**
```python
_input_fingerprint(inputs: dict, length: int, device: torch.device) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `set`, `all`, `isinstance`, `inputs.values`, `bool`, `(mask == 1).all`, `type`, `(grid > 0).all`, `(grid[:, 0] == 1).all` 외 7개
- 반환 표현: `{'completion_len': length, 'tensors': {k: {'shape': list(t.shape), 'dtype': str(t.dtype), 'sha256': tensor_sha256(t)} for k, t in sorted(inputs.items())}}`

**`_hashes` — L41**
```python
_hashes(model) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tensor_sha256`, `model.named_parameters`
- 반환 표현: `{name: tensor_sha256(p) for name, p in model.named_parameters()}`

**`_compare` — L45**
```python
_compare(expected: torch.Tensor, actual: torch.Tensor, name: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `bool`, `torch.isfinite(expected).all`, `torch.isfinite`, `torch.isfinite(actual).all`, `tensor_sha256`, `float`, `(expected - actual).abs().max`, `(expected - actual).abs`, `list` 외 1개
- 반환 표현: `{'shape': list(expected.shape), 'dtype': str(expected.dtype), 'max_abs': difference, 'expected_sha256': before, 'actual_sha256': after, 'bit_equal': True}`

**`_observed_logits` — L56**
```python
_observed_logits(bundle, inputs: dict, length: int, *, reference: bool) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bundle.model.get_base_model().lm_head.register_forward_hook`, `bundle.model.get_base_model`, `torch.no_grad`, `policy.reference_logprobs`, `policy.completion_logprobs`, `_require`, `bool`, `torch.cat`, `lp.detach().cpu`, `lp.detach` 외 1개
- 반환 표현: `(torch.cat(values, dim=1), lp.detach().cpu())`

**`_measure_reference` — L71**
```python
_measure_reference(bundle, fresh, probes: list[tuple[str, dict, int]], *, initial: bool) -> dict
```
- 기존 설명: Private tiny-model seam; never certifies a pretrained model or GPU run.
- 직접 호출: `bundle.model.get_base_model`, `_require`, `type`, `hasattr`, `len`, `all`, `isinstance`, `next`, `base.parameters`, `m.parameters` 외 31개
- 반환 표현: `{'max_abs': 0.0, 'frozen_sha_unchanged': True, 'requires_grad_restored': True, 'probe_count': len(rows), 'probes': rows, 'device': str(device), 'base_sha256': reference_binding, 'native …`

**`measure_reference` — L149**
```python
measure_reference(bundle, fresh, probes: list[tuple[str, dict, int]], protocol: dict, *, stage: str, engine=None) -> dict
```
- 기존 설명: Measure only INITIAL or POST_SYNC; the runner must bind both real stages.
- 직접 호출: `_require`, `measure_inventory`, `_pinned_config`, `next`, `bundle.model.parameters`, `all`, `fresh.parameters`, `require_idle_engine`, `str`, `_measure_reference` 외 2개
- 반환 표현: `result`

### D18.073 `fine_tuning/notebook_rl/gate_sequence.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_sequence.py) · SHA256 `bff0c44bdff6f9fb84fbf079c56098f7b7e45b6eeb68db7f5f19c601a5375f4d`

모듈의 기존 설명: Remaining live gate stages around the session's one shared RL update.

**`_record` — L17**
```python
_record(name, value, record, should_stop) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `record`, `should_stop`, `TimeoutError`
- 반환 표현: `value`
- 명시적 예외: `TimeoutError`; 전파되는 callee 예외는 별도.

**`_reference` — L24**
```python
_reference(bundle, preparation, probes, model_path, protocol, *, stage, engine=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `checkpoint.capture_rng`, `next`, `bundle.model.parameters`, `ValueError`, `launch.read_json`, `sum`, `row['path'].endswith`, `torch.cuda.mem_get_info`, `load_base(Path(model_path)).to`, `load_base` 외 9개
- 반환 표현: `gate_reference.measure_reference(bundle, fresh, inputs, protocol, stage=stage, engine=engine)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`before_engine` — L52**
```python
before_engine(bundle, preparation, probes, model_path, protocol, *, record, should_stop) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `should_stop`, `TimeoutError`, `_record`, `gpu_gates.measure_inventory`, `_reference`, `next`, `bundle.model.parameters`, `len`, `preparation.model_inputs`, `gate_gradients.measure_gradient_routes` 외 1개
- 반환 표현: `measured`
- 명시적 예외: `TimeoutError`; 전파되는 callee 예외는 별도.

**`_logprob_requests` — L71**
```python
_logprob_requests(probes, version, seed) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `replace`, `seeds`
- 반환 표현: `[replace(p.request, request_id=p.request.request_id + f':G7:{version}', n=16, seeds=seeds(p.request.request_id, version, 1, 'G7', 16, master_seed=seed)) for p in probes]`

**`before_update` — L77**
```python
before_update(bundle, engine, probes, protocol, runtime, *, seed, record, should_stop) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `should_stop`, `TimeoutError`, `gate_weights.require_idle_engine`, `ValueError`, `_record`, `gate_images.measure_image_parity`, `gate_logprobs.measure_logprobs`, `_logprob_requests`, `gate_tokens.measure_tokens`, `gate_cache.measure_current`
- 반환 표현: `measured`
- 명시적 예외: `TimeoutError`, `ValueError`; 전파되는 callee 예외는 별도.

**`after_update` — L100**
```python
after_update(bundle, engine, preparation, probes, model_path, protocol, *, seed, record, should_stop) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `should_stop`, `TimeoutError`, `ValueError`, `_record`, `gate_weights.measure_weight_parity`, `gate_cache.measure_current`, `gate_logprobs.measure_logprobs`, `_logprob_requests`, `_reference`
- 반환 표현: `measured`
- 명시적 예외: `TimeoutError`, `ValueError`; 전파되는 callee 예외는 별도.

**`combine` — L116**
```python
combine(measured, shared, cache, replay, protocol, scope) -> unannotated
```
- 기존 설명: Validate all observed components, without manufacturing approval receipts.
- 직접 호출: `all`, `ValueError`, `replay.get`, `set`, `values.update`, `_gpu_gates`, `values.items`
- 반환 표현: `values`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.074 `fine_tuning/notebook_rl/gate_session.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_session.py) · SHA256 `6bf55569439090d7357a231ad8059974f58e00179f04dd19f211a9cc3b821166`

모듈의 기존 설명: Bounded native gate sequence and fresh-process controls, not training approval.

**`_require` — L30**
```python
_require(value, message) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_actors` — L35**
```python
_actors(validated) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `from_json`, `_bytes(a).decode`, `_bytes`
- 반환 표현: `[a if type(a) is ActorItem else from_json(ActorItem, _bytes(a).decode()) for a in validated['actors']]`

**`_validation_hash` — L39**
```python
_validation_hash(validated) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_bytes`, `json.loads`, `to_json`, `_actors`
- 반환 표현: `_sha(_bytes(validated | {'actors': [json.loads(to_json(a)) for a in _actors(validated)]}))`

**`validate_job` — L43**
```python
validate_job(job: dict, protocol: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `set`, `job['runtime_scope'].get`, `syntax.update`, `measure.validate_job`, `launch.descriptor(job['sync_probes']).resolve`, `launch.descriptor`, `Path(job['output']).resolve`, `Path` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_probe_actors` — L63**
```python
_probe_actors(job, validated) -> unannotated
```
- 기존 설명: Probe images are eligible actors too; a caller cannot add arbitrary data.
- 직접 호출: `launch.read_descriptor`, `_require`, `set`, `type`, `len`, `json.loads`, `to_json`, `_actors`, `isinstance`, `bool` 외 8개
- 반환 표현: `result`

**`preflight` — L86**
```python
preflight(job: dict, protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_job`, `approval.validate_for_gate_data`, `Path`, `_probe_actors`
- 반환 표현: `result`

**`_eligible` — L95**
```python
_eligible(result) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `result.get`
- 반환 표현: `result.get('status') == 'VALIDATED_FOR_MEASUREMENT' and result.get('measurement_allowed') is True and (result.get('training_allowed') is False)`

**`_shared_step` — L100**
```python
_shared_step(bundle, optimizer, scheduler, engine, preparation, actors, structures, loader, verifier, prompts, protocol, probes, *, job, record, should_stop) -> unannotated
```
- 기존 설명: Observe one and the same production update; no extra G13 optimizer call.
- 직접 호출: `measure_guards`, `Path`, `measure_step`, `_require`, `record`
- 반환 표현: `{'G11': result, 'G13': guard}`

**`_capture` — L124**
```python
_capture(job, protocol, validated, runtime, output, binding, record) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `load_prompts`, `_require`, `_probe_actors`, `_actors`, `random.seed`, `np.random.seed`, `torch.manual_seed`, `record`, `prepare_vllm_schema_dependency` 외 33개
- 반환 표현: `{'status': 'CAPTURE_COMPLETED', 'input_packet': packet, 'capture': str(output / 'capture'), 'shared_sha256': _sha(_read(output / 'shared.json')), 'saved_step': 1, 'shared_checkpoint': s …`

**`worker` — L212**
```python
worker(task_descriptor: dict, protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `launch.read_descriptor`, `_require`, `set`, `preflight`, `_eligible`, `_validation_hash`, `launch.checkpoint_binding`, `previous.get`, `launch.runtime_check`, `Path` 외 25개
- 반환 표현: `result`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`_run_child` — L321**
```python
_run_child(task: dict, path: Path, deadline: float) -> subprocess.Popen
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_bytes`, `_write`, `_require`, `time.time`, `path.with_suffix('.log').open`, `path.with_suffix`, `subprocess.Popen`, `str`, `_sha`, `child.wait` 외 2개
- 반환 표현: `child`

**`execute` — L342**
```python
execute(job: dict, protocol: dict, validated: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `torch.cuda.is_initialized`, `_eligible`, `preflight`, `Path`, `_safe_root`, `output.mkdir`, `_validation_hash`, `str`, `_run_child` 외 14개
- 반환 표현: `result`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`main` — L408**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_mutually_exclusive_group`, `mode.add_argument`, `parser.add_argument`, `parser.parse_args`, `signal.signal`, `load_protocol`, `_require`, `worker`, `str` 외 9개
- 반환 표현: `0 if _eligible(checked) else 2`; `0`; `2`

### D18.075 `fine_tuning/notebook_rl/gate_snapshot.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_snapshot.py) · SHA256 `143e99c36b271331676d3554731392695b321412457c4f4832c22bcb43715d87`

모듈의 기존 설명: File-bound step1 exports for isolated G6 cache controls; no GPU allocation.

**`_require` — L16**
```python
_require(value: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_hash` — L21**
```python
_hash(value) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `re.fullmatch`
- 반환 표현: `isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None`

**`validate_snapshot` — L25**
```python
validate_snapshot(descriptor: dict, binding: dict, checkpoint_path: Path) -> dict
```
- 기존 설명: Validate the immutable export against its actual saved step1 source.
- 직접 호출: `launch.descriptor`, `_safe_root`, `_require`, `launch.read_descriptor`, `Path`, `checkpoint.read_metadata`, `manifest.get`, `checkpoint.file_hash`, `ckpt.iterdir`, `type` 외 14개
- 반환 표현: `manifest`

**`export_step1` — L73**
```python
export_step1(captured: dict, model_path: Path, output: Path, arm: str, binding: dict) -> dict
```
- 기존 설명: Reload saved step1 on CPU, never export the caller's live step2 policy.
- 직접 호출: `_require`, `torch.cuda.is_initialized`, `captured.get`, `isinstance`, `Path(captured['shared_checkpoint']).is_absolute`, `Path`, `_safe_root`, `checkpoint.read_metadata`, `Path(output).absolute`, `export.export_merged` 외 3개
- 반환 표현: `descriptor`

**`iter_tensors` — L91**
```python
iter_tensors(descriptor: dict, binding: dict, checkpoint_path: Path) -> Iterator[tuple[str, torch.Tensor]]
```
- 기존 설명: Stream each canonical CPU BF16 tensor once; exhaust before accepting a load.
- 직접 호출: `validate_snapshot`, `launch.descriptor`, `set`, `sorted`, `name.endswith`, `safe_open`, `shard.keys`, `_require`, `shard.get_tensor`, `list` 외 5개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.076 `fine_tuning/notebook_rl/gate_throughput.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_throughput.py) · SHA256 `7463049d1eb641760e9857861801ac40464d6e02b1edb3c86eaff52985d061b3`

모듈의 기존 설명: G11: one measured production rollout/update/synchronization, not training approval.

**`_require` — L24**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_hash` — L29**
```python
_hash(value) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).e …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()`

**`_input_row` — L34**
```python
_input_row(prepared, images) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `set`, `isinstance`, `list`, `len`, `bool`, `(grid > 0).all`, `(grid[:, 0] == 1).all`, `grid.tolist`, `int` 외 9개
- 반환 표현: `{'image_count': len(images), 'image_grid_thw': grid.tolist(), 'patch_cells': int(grid.prod(-1).sum()), 'prompt_tokens': len(prepared.token_ids), 'prompt_sha256': _hash(prepared.token_id …`

**`_check_input` — L53**
```python
_check_input(prepared, images, expected, *, initial) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_input_row`, `observed.pop`, `_require`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_native_budgets` — L62**
```python
_native_budgets(engine, protocol) -> dict
```
- 기존 설명: Read resolved native configuration, not wrapper constructor arguments.
- 직접 호출: `dict`, `expected.update`, `_require`, `all`, `type`
- 반환 표현: `actual`

**클래스 `_Incomplete` — L90**

기저: `Exception`.

**`_cohort` — L94**
```python
_cohort(actors, structures, loader, engine, prompts, cfg, should_stop, record) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `bool`, `set`, `all`, `actors.items`, `sorted`, `should_stop`, `_Incomplete`, `loader`, `len` 외 12개
- 반환 표현: `(rows, order)`
- 명시적 예외: `_Incomplete`; 전파되는 callee 예외는 별도.

**클래스 `_Memory` — L124**

기저: `object`.

**`_Memory.__init__` — L126**
```python
_Memory.__init__(self, device) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `threading.Event`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_Memory._sample` — L134**
```python
_Memory._sample(self) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.cuda.mem_get_info`, `_require`, `max`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_Memory._poll` — L143**
```python
_Memory._poll(self) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.stop.wait`, `self._sample`, `type`
- 반환 표현: `None`

**`_Memory.start` — L151**
```python
_Memory.start(self) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.cuda.synchronize`, `torch.cuda.reset_peak_memory_stats`, `self._sample`, `threading.Thread`, `self.thread.start`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_Memory.finish` — L159**
```python
_Memory.finish(self) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.stop.set`, `self.thread.join`, `_require`, `self.thread.is_alive`, `self._sample`, `str`, `torch.cuda.max_memory_allocated`, `torch.cuda.max_memory_reserved`
- 반환 표현: `{'device_sampled_peak_bytes': None, 'torch_peak_allocated_bytes': None, 'torch_peak_reserved_bytes': None, 'device_samples': 0, 'device_total_bytes': None, 'memory_observed': False}`; `{'device_sampled_peak_bytes': self.peak, 'device_samples': self.count, 'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device), 'torch_peak_reserved_bytes': torch.cud …`

**클래스 `_TimedSampler` — L176**

기저: `object`.

**`_TimedSampler.__init__` — L177**
```python
_TimedSampler.__init__(self, engine, rows, device, record) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_TimedSampler.__getattr__` — L182**
```python
_TimedSampler.__getattr__(self, name) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `getattr`
- 반환 표현: `getattr(self.engine, name)`

**`_TimedSampler.prepare` — L185**
```python
_TimedSampler.prepare(self, request) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.engine.prepare`, `request.request_id.rsplit`, `_require`, `_check_input`
- 반환 표현: `prepared`

**`_TimedSampler.sample` — L192**
```python
_TimedSampler.sample(self, requests) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.prepare`, `_stage_time`, `self.engine.sample`, `self.record`, `asdict`, `sum`, `len`
- 반환 표현: `results`

**`_measure_step` — L210**
```python
_measure_step(bundle, optimizer, scheduler, engine, preparation, actors, structures, loader, verifier, prompts, protocol, probes, *, algorithm, step, seed, should_stop, record, _native=False) -> dict
```
- 기존 설명: Private CPU seam uses real tiny gradients but is never native/GPU evidence.
- 직접 호출: `RolloutConfig.from_protocol`, `cfg.validate`, `_require`, `next`, `bundle.model.parameters`, `all`, `_Memory`, `_cohort`, `result.update`, `_hash` 외 29개
- 반환 표현: `result`
- 명시적 예외: `_Incomplete`, `re-raise`; 전파되는 callee 예외는 별도.

**`measure_step` — L327**
```python
measure_step(bundle, optimizer, scheduler, engine, preparation, actors, structures, loader, verifier, prompts, protocol, probes, *, algorithm, step, seed, should_stop, record) -> dict
```
- 기존 설명: Live native G11 stress: two images + actual third attempt, one update + sync.
- 직접 호출: `next`, `bundle.model.parameters`, `_require`, `type`, `_pinned_config`, `measure_inventory`, `require_idle_engine`, `_native_budgets`, `str`, `record` 외 2개
- 반환 표현: `result`

### D18.077 `fine_tuning/notebook_rl/gate_tokens.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_tokens.py) · SHA256 `642c5e7bf6f662674fd9cf0a324358865879fb503f0808d4d09f89d8441deebf`

모듈의 기존 설명: G8 live two-turn token-boundary measurement, not a correctness experiment.

**`_require` — L20**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_pixels` — L25**
```python
_pixels(prepared) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `str`, `tensor_sha256`, `prepared.pixels.items`
- 반환 표현: `{'image_key': prepared.image_key, 'image_spans': [list(s) for s in prepared.image_spans], 'tensors': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': tensor_sha25 …`

**`_sample` — L31**
```python
_sample(engine, request: RolloutRequest, version: int, turn: int, parent: Sample | None, record: Callable[[dict], None]) -> tuple[Sample, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `engine.prepare`, `list`, `_pixels`, `engine.sample`, `record`, `type`, `asdict`, `_require`, `len`, `_completion` 외 15개
- 반환 표현: `(sample, row)`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`_measure_tokens` — L84**
```python
_measure_tokens(engine, requests: list[RolloutRequest], *, feedback: str, retry_max_new_tokens: int, record: Callable[[dict], None]) -> dict
```
- 기존 설명: Private seam also permits real tiny-HF CPU tests; never native evidence.
- 직접 호출: `_require`, `callable`, `isinstance`, `bool`, `feedback.strip`, `type`, `len`, `all`, `_request`, `tokenizer.convert_tokens_to_ids` 외 10개
- 반환 표현: `{'tokens_exact': True, 'generated_only_mask': True, 'eos_checked': True, 'policy_version': version, 'probes': rows, 'actual_calls': 4, 'conversation_boundary_token_id': end, 'closure_br …`

**`measure_tokens` — L140**
```python
measure_tokens(engine, requests: list[RolloutRequest], *, feedback: str, retry_max_new_tokens: int, record: Callable[[dict], None]) -> dict
```
- 기존 설명: Run four actual unscored requests on an already-owned native engine.
- 직접 호출: `require_idle_engine`, `_pinned_config`, `_measure_tokens`, `_require`, `result.update`
- 반환 표현: `result`

### D18.078 `fine_tuning/notebook_rl/gate_update.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_update.py) · SHA256 `814d30ff8cc3184af3021e2bb8c924f6b759725238be91fef549a126d2432270`

모듈의 기존 설명: G4 evidence around one real reward-based update of a caller-owned policy.

**`_require` — L17**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_base_hashes` — L22**
```python
_base_hashes(bundle) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tensor_sha256`, `policy._canonical_frozen`, `_require`
- 반환 표현: `result`

**`_masters` — L28**
```python
_masters(bundle) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `checkpoint._parameters`, `_require`, `len`, `id`, `rows.items`, `name.removeprefix('base_model.model.').startswith`, `name.removeprefix`, `probe_indices`, `p.numel`, `list` 외 9개
- 반환 표현: `result`

**`_measure_update` — L50**
```python
_measure_update(bundle, optimizer, scheduler, trees, images, preparation, protocol: dict, algorithm: str, *, step: int, seed: int, vision_disabled: bool=False) -> dict
```
- 기존 설명: Private actual tiny-CPU seam; never a pretrained-model/GPU certificate.
- 직접 호출: `_require`, `type`, `torch.is_inference_mode_enabled`, `torch.is_grad_enabled`, `dict`, `bundle.model.named_parameters`, `bool`, `all`, `named.values`, `len` 외 29개
- 반환 표현: `{'merger_changed': bool(changed) and bool(probe_changed), 'frozen_unchanged': base_after == base_before, 'status': 'UPDATED_UNSYNCHRONIZED', 'update': update, 'engine_sync_required': Tr …`

**`measure_update` — L122**
```python
measure_update(bundle, optimizer, scheduler, trees, images, preparation, protocol: dict, algorithm: str, *, step: int, seed: int, vision_disabled: bool=False) -> dict
```
- 기존 설명: Perform and measure ONE real update on a live pinned CUDA policy, without sync.
- 직접 호출: `_require`, `type`, `bundle.model.parameters`, `len`, `next`, `iter`, `_pinned_config`, `measure_inventory`, `_measure_update`, `result.update`
- 반환 표현: `result`

### D18.079 `fine_tuning/notebook_rl/gate_weights.py`

[현재 소스](../../fine_tuning/notebook_rl/gate_weights.py) · SHA256 `8e2c57b25509e8db46acfd933680f61fda2f543467bd08a26be402b3d3d8cf7b`

모듈의 기존 설명: G5 complete live tensor readback for pinned, unquantized TP1 Qwen3-VL.

**`_require` — L39**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`require_idle_engine` — L44**
```python
require_idle_engine(engine, *, require_push: bool=True) -> dict
```
- 기존 설명: Shared live-engine guard; fresh F0 is allowed only without a push claim.
- 직접 호출: `_require`, `type`, `version`, `isinstance`, `import_module`, `torch.device`, `getattr`, `all`, `topology.values`, `scheduler.has_unfinished_requests` 외 3개
- 반환 표현: `{'vllm_version': '0.11.0', 'in_process': True, 'topology': topology, 'device': str(device), 'dtype': 'bfloat16', 'quantization': None, 'policy_version': policy_version, 'last_loaded_ver …`

**`_layout` — L82**
```python
_layout(shapes: dict[str, tuple[int, ...]]) -> dict
```
- 기존 설명: Independent HF-prefix and dim-0 packing map; not the native weight loader.
- 직접 호출: `sorted`, `shapes.items`, `_require`, `bool`, `all`, `type`, `name.startswith`, `name.removeprefix`, `ValueError`, `re.fullmatch` 외 9개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_read_worker` — L122**
```python
_read_worker(model, layout: dict, *, native: bool) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `type`, `dict`, `model.named_parameters`, `set`, `all_parameters.items`, `aliases.setdefault(id(parameter), []).append`, `aliases.setdefault`, `id`, `sorted` 외 18개
- 반환 표현: `{'parameters': rows, 'aliases': alias_groups, 'device': next(iter(devices))}`

**`_measure_parity` — L156**
```python
_measure_parity(named: Iterable[tuple[str, torch.Tensor]], shapes: dict[str, tuple[int, ...]], apply_model: Callable, *, native: bool=False) -> dict
```
- 기존 설명: Private authored-worker seam; its default never certifies native execution.
- 직접 호출: `_layout`, `_require`, `tuple`, `bool`, `torch.isfinite(tensor).all`, `torch.isfinite`, `list`, `str`, `tensor_sha256`, `set` 외 9개
- 반환 표현: `{'all_tensors_match': all((r['expected_sha256'] == r['actual_sha256'] for r in ledger)), 'all_four_mergers': len(merger_names) == 24 and merger_names <= {r['canonical_name'] for r in le …`

**`_pinned_config` — L193**
```python
_pinned_config(config) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `getattr`, `dict`, `list`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`measure_weight_parity` — L209**
```python
measure_weight_parity(bundle, engine) -> dict
```
- 기존 설명: Read actual post-push worker tensors; no report grants training permission.
- 직접 호출: `require_idle_engine`, `_require`, `_pinned_config`, `len`, `set`, `dict`, `assert_base_unchanged`, `_measure_parity`, `merged_named_tensors`, `report.update`
- 반환 표현: `report`

### D18.080 `fine_tuning/notebook_rl/generators/__init__.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/__init__.py) · SHA256 `19c9fbe4c69c7f9b199e29003dc1687c1678947ac83a2eec464017d69f9a45a7`

모듈의 기존 설명: Load generators only when requested; data dependencies stay out of training.

**`get_generator` — L10**
```python
get_generator(name: str) -> ModuleType
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `import_module`, `all`, `hasattr`, `RuntimeError`
- 반환 표현: `REGISTRY[name]`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

### D18.081 `fine_tuning/notebook_rl/generators/chart.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/chart.py) · SHA256 `75b9d695d3d511db5a2e307c903833ef8eb083543c81b9fad2cb1718b46c41b9`

모듈의 기존 설명: Authored financial charts with exact dual solvers; records are not training-approved.

**`_select` — L25**
```python
_select(index: int, seed: int, split: Split, style: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `matches_split`, `random.Random`, `range`, `rng.randint`, `rng.choice`, `values.append`, `values.extend`, `list` 외 2개
- 반환 표현: `structure`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`_array_answer` — L45**
```python
_array_answer(structure: dict, task: str) -> Fraction | str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Fraction`, `abs`, `zip`, `max`, `gaps.index`, `ValueError`
- 반환 표현: `Fraction(100 * (revenue[1] - revenue[0]), revenue[0])`; `Fraction(100 * (revenue[0] - cost[0]), revenue[0])`; `Fraction(max(gaps), structure['scale'])`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_pandas_answer` — L59**
```python
_pandas_answer(structure: dict, task: str) -> Fraction | str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `pd.DataFrame`, `Fraction`, `zip`, `frame['Revenue'].pct_change`, `frame['Cost'].div`, `frame['Revenue'].sub(frame['Cost']).abs`, `frame['Revenue'].sub`, `gaps.max`, `str`, `gaps.idxmax` 외 1개
- 반환 표현: `frame['Revenue'].pct_change(fill_method=None).iloc[1] * 100`; `(1 - frame['Cost'].div(frame['Revenue'])).iloc[0] * 100`; `gaps.max()`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_decimal` — L77**
```python
_decimal(value: Fraction) -> str
```
- 기존 설명: Nearest tenth, with exact rational halfway cases rounded away from zero.
- 직접 호출: `divmod`, `abs`
- 반환 표현: `f'{sign}{quotient // 10}.{quotient % 10}'`

**`_value` — L85**
```python
_value(ticks: int, scale: int) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`
- 반환 표현: `str(ticks) if scale == 1 else f'{ticks // 10}.{ticks % 10}'`

**`_question` — L89**
```python
_question(task: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `'Which category has the largest absolute difference between Revenue and Cost? If tied, choose the earliest category in the displayed order. Answer with its category label.'`; `question + ' Round to one decimal place; round exact halfway cases away from zero.'`

**`_choices` — L105**
```python
_choices(answer: Fraction, structure: dict, seed: int) -> tuple[tuple[str, ...], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Fraction`, `_decimal`, `options.append`, `random.Random`, `len`, `rng.choice`, `rng.randint`, `rng.shuffle`, `tuple`, `chr` 외 1개
- 반환 표현: `(tuple(options), chr(65 + options.index(correct)))`

**`_slot` — L129**
```python
_slot(family: str, series: str, item: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `f'bar[{series}][{item}]'`; `f'point[{series}][{item}]'`; `f'cell[{item}][{series}]'`

**`_render` — L137**
```python
_render(structure: dict, family: str, style: str) -> tuple[bytes, tuple[str, ...]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Figure`, `FigureCanvasAgg`, `figure.add_subplot`, `ax.set_facecolor`, `ax.set_title`, `_value`, `ax.set_axis_off`, `range`, `len`, `ax.table` 외 31개
- 반환 표현: `(png_bytes(figure, dpi=110 if shifted else 100), tuple(printed))`

**`generate` — L205**
```python
generate(index: int, seed: int, split: Split, mode: Mode, style: str='default') -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `_select`, `_array_answer`, `_pandas_answer`, `isinstance`, `_decimal`, `_choices`, `int`, `structure_hash`, `enumerate` 외 7개
- 반환 표현: `build(source=GENERATOR_VERSION, family=family, index=index, split=split, mode=mode, question=_question(task), choices=choices, answer_type=answer_type, gold=gold, accept={}, slots=slots …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`self_check` — L240**
```python
self_check(p: GeneratedProblem) -> None
```
- 기존 설명: Recompute both solvers and rerender, then check the complete immutable record.
- 직접 호출: `isinstance`, `ValueError`, `int`, `p.actor.problem_id.split`, `p.lineage.image_provenance.rsplit`, `generate`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.082 `fine_tuning/notebook_rl/generators/chem.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/chem.py) · SHA256 `6318bfeda2619323dd8af1daeda30b47045a73a41d21c07fc4a4eba7e591dee7`

모듈의 기존 설명: ChEBI-SMILES rendering and program rewards; no download or training approval.

**`_source` — L32**
```python
_source(raw: bytes) -> tuple[tuple[str, str], ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `csv.DictReader`, `io.StringIO`, `raw.decode`, `ValueError`, `set`, `isinstance`, `re.fullmatch`, `ids.add`, `Chem.MolFromSmiles`, `mol.GetNumHeavyAtoms` 외 14개
- 반환 표현: `tuple(sorted(((ident, smiles) for smiles, ident in eligible.items())))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_load_source` — L64**
```python
_load_source() -> tuple[bytes, tuple[tuple[str, str], ...]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `os.environ.get`, `ValueError`, `Path(path).read_bytes`, `Path`, `_source`
- 반환 표현: `(raw, _source(raw))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_counts` — L72**
```python
_counts(mol) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Counter`, `mol.GetAtoms`, `atom.GetSymbol`, `atom.GetAtomicNum`, `atom.GetTotalNumHs`, `counts.items`
- 반환 표현: `{k: v for k, v in counts.items() if v}`

**`_manual_formula` — L81**
```python
_manual_formula(mol) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_counts`, `sorted`, `set`, `''.join`, `str`
- 반환 표현: `''.join((element + (str(counts[element]) if counts[element] != 1 else '') for element in order if element in counts))`

**`_cycle_rank` — L90**
```python
_cycle_rank(mol) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `range`, `mol.GetNumAtoms`, `mol.GetBonds`, `root`, `bond.GetBeginAtomIdx`, `bond.GetEndAtomIdx`
- 반환 표현: `cycles`

**`_manual_donors` — L106**
```python
_manual_donors(mol) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `mol.GetAtoms`, `atom.GetAtomicNum`, `atom.GetTotalNumHs`, `atom.GetTotalValence`, `atom.GetIsAromatic`
- 반환 표현: `count`

**`_manual_carboxyl` — L120**
```python
_manual_carboxyl(mol) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `mol.GetAtoms`, `atom.GetAtomicNum`, `atom.GetIsAromatic`, `atom.GetBonds`, `bond.GetOtherAtom`, `neighbor.GetAtomicNum`, `bond.GetBondType`, `neighbor.GetTotalNumHs`, `bool`
- 반환 표현: `count`

**`_answers` — L135**
```python
_answers(mol, task: str) -> tuple[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `rdMolDescriptors.CalcMolFormula`, `_manual_formula`, `str`, `len`, `Chem.GetSSSR`, `_cycle_rank`, `rdMolDescriptors.CalcNumHBD`, `_manual_donors`, `Chem.MolFromSmarts`, `mol.GetSubstructMatches` 외 2개
- 반환 표현: `(rdMolDescriptors.CalcMolFormula(mol), _manual_formula(mol))`; `(str(len(Chem.GetSSSR(mol))), str(_cycle_rank(mol)))`; `(str(rdMolDescriptors.CalcNumHBD(mol)), str(_manual_donors(mol)))`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_render` — L148**
```python
_render(mol, style: str) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Chem.Mol`, `rdDepictor.Compute2DCoords`, `rdMolDraw2D.MolDraw2DSVG`, `drawer.drawOptions`, `options.useBWAtomPalette`, `options.setBackgroundColour`, `drawer.DrawMolecule`, `drawer.FinishDrawing`, `cairosvg.svg2png`, `drawer.GetDrawingText().encode` 외 4개
- 반환 표현: `png_bytes(image, dpi=100)`

**`generate` — L168**
```python
generate(index: int, seed: int, split: Split, mode: Mode, style: str='default') -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `matches_split`, `_load_source`, `random.Random`, `range`, `rng.randrange`, `len`, `structure_hash`, `RuntimeError` 외 14개
- 반환 표현: `replace(p, lineage=replace(p.lineage, license='CC-BY-4.0 (ChEBI source; review required)', upstream_origin='ChEBI', parents=(ident, 'chebi_tsv_sha256:' + hashlib.sha256(raw).hexdigest() …`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`self_check` — L205**
```python
self_check(p: GeneratedProblem) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `int`, `p.actor.problem_id.split`, `p.lineage.image_provenance.rsplit`, `generate`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.083 `fine_tuning/notebook_rl/generators/circuit.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/circuit.py) · SHA256 `4482ee658a178cb627f4974fe9afb419053a9cd11be0ac983557a3f9e0afeb67`

모듈의 기존 설명: Exact program-authored DC bridge and 2–3-stage ladders; no AC/transient scope.

**`_validate` — L25**
```python
_validate(world: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `ValueError`, `circuit_bridge.validate_world`, `type`, `len`, `any`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_components` — L42**
```python
_components(world: dict) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `circuit_bridge.canonical_structure`, `result.append`, `str`, `comp.get`, `len`, `enumerate`, `zip`, `result.extend`
- 반환 표현: `result`

**`_ladder_nodal` — L65**
```python
_ladder_nodal(world: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `Fraction`, `sum`, `range`, `_components`, `int`, `unknown.index`, `ValueError`, `zip`, `nodes.update`
- 반환 표현: `{'node_voltages_V': nodes, 'resistor_currents_A': currents, 'source_current_A': current, 'equivalent_R_ohm': Fraction(sum(world['sources'])) / current}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_ladder_reduction` — L102**
```python
_ladder_reduction(world: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `map`, `range`, `len`, `Fraction`, `sum`, `enumerate`, `zip`, `ValueError`
- 반환 표현: `{'node_voltages_V': nodes, 'resistor_currents_A': currents, 'source_current_A': current, 'equivalent_R_ohm': resistance}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_solutions` — L125**
```python
_solutions(world: dict) -> tuple[dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_validate`, `_ladder_nodal`, `_ladder_reduction`, `circuit_bridge.solve_nodal`, `circuit_bridge.verify_series_parallel`, `convert`
- 반환 표현: `(_ladder_nodal(world), _ladder_reduction(world))`; `(convert(first), convert(second))`

**`_query` — L140**
```python
_query(world: dict, index: int) -> tuple[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`
- 반환 표현: `(kind, 'R2' if world['kind'] == 'bridge' else 'R4')`; `(kind, 'N2' if world['kind'] == 'bridge' else f"N{len(world['sources']) + len(world['series'])}")`; `(kind, 'input')`

**`_answer` — L149**
```python
_answer(solution: dict, query: tuple[str, str]) -> tuple[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `(f"{solution['resistor_currents_A'][target]} A", 'A')`; `(f"{solution['node_voltages_V'][target]} V", 'V')`; `(f"{solution['equivalent_R_ohm']} ohm", 'ohm')`

**`_question` — L158**
```python
_question(query: tuple[str, str]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `base + f'What is the downward current through {target}? Give an exact integer or reduced fraction with unit A.'`; `base + f'What is the voltage of {target} relative to N0? Give an exact integer or reduced fraction with unit V.'`; `base + 'What equivalent resistance does the resistive load present to the source(s)? Give an exact integer or reduced fraction with unit ohm.'`

**`_facts` — L170**
```python
_facts(world: dict) -> tuple[dict[str, str], dict[str, str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_components`, `dict`, `sorted`, `facts.items`, `types.items`
- 반환 표현: `(dict(sorted(facts.items())), dict(sorted(types.items())))`

**`_render` — L184**
```python
_render(world: dict, style: str) -> tuple[bytes, tuple[str, ...]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `ImageFont.load_default_imagefont`, `ImageFont.load_default`, `label`, `len`, `range`, `positions.update`, `positions.items`, `positions.values` 외 10개
- 반환 표현: `(png_bytes(image, dpi=100), tuple(printed))`

**`generate` — L248**
```python
generate(index: int, seed: int, split: Split, mode: Mode, style: str='default') -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `matches_split`, `random.Random`, `range`, `rng.randint`, `bool`, `rng.randrange`, `structure_hash`, `_solutions` 외 10개
- 반환 표현: `build(source=GENERATOR_VERSION, family=FAMILY, index=index, seed=seed, split=split, mode=mode, style=style, question=_question(query), choices=(), answer_type='unit_number', gold=gold,  …`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`self_check` — L286**
```python
self_check(p: GeneratedProblem) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `tuple`, `set`, `len`, `_query`, `ValueError`, `_solutions`, `_facts`, `_render`, `_answer` 외 6개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.084 `fine_tuning/notebook_rl/generators/common.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/common.py) · SHA256 `57b50be6a9bd802b1f57c0c721084028065155188a3ac3dae8ecaf6ea7f25748`

모듈의 기존 설명: Deterministic, unapproved generated records; no filesystem or model writes.

**클래스 `GeneratedProblem` — L20**

기저: `object`.

```python
actor: ActorItem
verifier: VerifierItem
lineage: Lineage
image_bytes: tuple[bytes, ...]
```

**`structure_hash` — L27**
```python
structure_hash(obj: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `check`, `json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False …`, `json.dumps`, `ValueError`, `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`
- 반환 표현: `hashlib.sha256(raw).hexdigest()[:16]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`png_chunks` — L48**
```python
png_chunks(raw: bytes) -> tuple[bytes, ...]
```
- 기존 설명: Check chunk bounds and CRCs as well as the metadata allowlist.
- 직접 호출: `type`, `raw.startswith`, `ValueError`, `len`, `struct.unpack`, `zlib.crc32`, `kinds.append`, `Image.open`, `io.BytesIO`, `image.verify` 외 1개
- 반환 표현: `tuple(kinds)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`png_bytes` — L79**
```python
png_bytes(fig_or_image: Any, *, dpi: int) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `io.BytesIO`, `isinstance`, `fig_or_image.savefig`, `Image.open`, `source.getvalue`, `picture.getbands`, `picture.convert`, `pixels.info.clear` 외 4개
- 반환 표현: `raw`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`split_of` — L105**
```python
split_of(structure_hash: str, ratios=(0.8, 0.1, 0.1)) -> Split
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `re.fullmatch`, `ValueError`, `len`, `any`, `type`, `math.isfinite`, `math.isclose`, `sum`, `int`
- 반환 표현: `'train' if value < ratios[0] else 'dev' if value < sum(ratios[:2]) else 'sealed'`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`matches_split` — L115**
```python
matches_split(digest: str, split: Split, style: str='default') -> bool
```
- 기존 설명: Reserve half the sealed hash band for unseen shifted-style structures.
- 직접 호출: `ValueError`, `split_of`, `int`
- 반환 표현: `split == base`; `split == held`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`build` — L128**
```python
build(source: str, family: str, index: int, split: Split, mode: Mode, question: str, choices, answer_type: AnswerType, gold: str, accept: dict, slots: dict[str, str], slot_types: dict[str, str], required_slots, oracle_fact_lines, printed_text, structure, template_id: str, image_bytes, generator_version: str, seed: int, style: str) -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `all`, `isinstance`, `re.fullmatch`, `ValueError`, `type`, `structure_hash`, `matches_split`, `tuple`, `len`, `png_chunks` 외 9개
- 반환 표현: `GeneratedProblem(actor, verifier, lineage, images)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.085 `fine_tuning/notebook_rl/generators/graph.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/graph.py) · SHA256 `ffa8a1ffb17c06932197a1fbf91f4f731222838da4de85158fd45ee083e98dd5`

모듈의 기존 설명: Bounded program-authored graph/CS problems; no benchmark, teacher, or CPM inputs.

**`_network` — L30**
```python
_network(world: dict, *, undirected: bool=False) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `nx.Graph`, `nx.DiGraph`, `graph.add_nodes_from`, `graph.add_weighted_edges_from`
- 반환 표현: `graph`

**`_dijkstra` — L37**
```python
_dijkstra(world: dict) -> int | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `heapq.heappop`, `heapq.heappush`
- 반환 표현: `cost`; `None`

**`_bfs` — L52**
```python
_bfs(world: dict) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `deque`, `pending.popleft`, `order.append`, `sorted`, `seen.add`, `pending.append`
- 반환 표현: `order`

**`_tree_order` — L64**
```python
_tree_order(world: dict, traversal: str, *, iterative: bool) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `stack.pop`, `result.append`, `children.get(node, {}).get`, `children.get`, `events.insert`, `stack.extend`, `reversed`, `visit`
- 반환 표현: `result`; `visit(world['root'])`

**`_answers` — L88**
```python
_answers(world: dict, traversal: str='postorder') -> tuple[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`, `_dijkstra`, `nx.shortest_path_length`, `_network`, `_bfs`, `list`, `nx.bfs_tree`, `_NODES.index`, `nvw.solve_signal`, `nvw.verify_signal` 외 11개
- 반환 표현: `(str(_dijkstra(world)), str(nx.shortest_path_length(_network(world), 'A', 'H', weight='weight')))`; `(','.join(first), ','.join(second))`; `('accept' if current in world['accepting'] else 'reject', 'accept' if final else 'reject')`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_shortcut_reason` — L117**
```python
_shortcut_reason(world: dict, traversal: str='postorder') -> str | None
```
- 기존 설명: Task-specific audits: graph_bridge.exclusion_reason only accepts signal-count episodes.
- 직접 호출: `_network`, `nx.has_path`, `len`, `nx.shortest_path`, `nx.shortest_path_length`, `_bfs`, `sorted`, `set`, `nx.node_connected_component`, `visited.add` 외 2개
- 반환 표현: `'UNREACHABLE_TARGET'`; `'ONE_EDGE_SHORTCUT'`; `'DIRECTION_FREE_SHORTCUT'`; 외 5개 분기

**`_sample` — L153**
```python
_sample(kind: str, rng: random.Random) -> dict | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `rng.random`, `rng.randrange`, `edges.append`, `rng.randint`, `nx.DiGraph`, `graph.add_nodes_from`, `graph.add_weighted_edges_from`, `dict`, `nx.all_pairs_dijkstra_path_length` 외 15개
- 반환 표현: `None`; `{'kind': kind, 'nodes': list(_NODES), 'edges': sorted(([labels[a], labels[b], weight] for a, b, weight in edges)), 'positions': {labels[i]: i for i in range(8)}}`; `{'kind': kind, 'start': 'q0', 'accepting': sorted(rng.sample(states, rng.randint(1, 3))), 'trans': {state: {symbol: rng.choice(states) for symbol in '01'} for state in states}}`; 외 1개 분기

**`_facts` — L207**
```python
_facts(world: dict) -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`, `','.join`, `world['trans'].items`, `trans.items`, `world['children'].items`, `children.items`, `dict`, `sorted`, `facts.items`
- 반환 표현: `dict(sorted(facts.items()))`

**`_slot_types` — L220**
```python
_slot_types(slots: dict[str, str]) -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `key.startswith`
- 반환 표현: `{key: 'number' if key.startswith('edge[') else 'text' if key == 'accepting' else 'relation' for key in slots}`

**`_validate_world` — L225**
```python
_validate_world(world: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `list`, `any`, `type`, `world['positions'].values`, `sorted`, `range`, `ValueError`, `frozenset`, `abs` 외 15개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_question` — L273**
```python
_question(kind: str, traversal: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `'What is the shortest path length from A to H in this directed weighted graph? Follow arrowheads and sum the weights. Crossings are not junctions. Give an integer.'`; `'Starting at A, list reachable vertices in BFS discovery order, including A. Follow outgoing arrows only; enqueue unseen neighbors in ascending alphabetical order. Omit unreachable vert …`; `'Does this DFA accept the word 0110? Read symbols left to right from the marked start state; double circles mark accepting states. Respond accept or reject. In observations, list accept …`; 외 1개 분기

**`_render` — L292**
```python
_render(world: dict, style: str) -> tuple[bytes, tuple[str, ...]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `ImageFont.load_default_imagefont`, `ImageFont.load_default`, `label`, `sorted`, `world['positions'].items`, `arrow`, `draw.rectangle`, `image.getpixel` 외 11개
- 반환 표현: `(png_bytes(image, dpi=100), tuple(printed))`

**`generate` — L372**
```python
generate(index: int, seed: int, split: Split, mode: Mode, style: str='default') -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `matches_split`, `random.Random`, `range`, `_sample`, `_validate_world`, `_shortcut_reason`, `world.items`, `structure_hash` 외 9개
- 반환 표현: `p`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`self_check` — L413**
```python
self_check(p: GeneratedProblem) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `_validate_world`, `template.rsplit`, `world.items`, `structure_hash`, `matches_split`, `_shortcut_reason`, `ValueError`, `_answers`, `_render` 외 8개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.086 `fine_tuning/notebook_rl/generators/graph_pair.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/graph_pair.py) · SHA256 `546e9eb5b4f849e1713600c3b35b7aaee1dc90f8365c9fe45dbaf8ed106d9a1b`

모듈의 기존 설명: Program-authored shortest paths split across topology and edge-weight images.

**`_render` — L24**
```python
_render(world: dict, style: str) -> tuple[tuple[bytes, bytes], tuple[str, ...]]
```
- 기존 설명: The topology never prints weights; the table never prints vertex endpoints.
- 직접 호출: `graph._validate_world`, `ValueError`, `ImageFont.load_default`, `Image.new`, `ImageDraw.Draw`, `label`, `sorted`, `world['positions'].items`, `enumerate`, `math.hypot` 외 11개
- 반환 표현: `((png_bytes(topology, dpi=100), png_bytes(table, dpi=100)), tuple(printed))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_record` — L73**
```python
_record(world: dict, index: int, seed: int, split: Split, mode: Mode, style: str) -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `graph._validate_world`, `graph._shortcut_reason`, `ValueError`, `graph._answers`, `_render`, `enumerate`, `str`, `world.items`, `build`, `key.startswith` 외 2개
- 반환 표현: `build(source=GENERATOR_VERSION, family=FAMILY, index=index, split=split, mode=mode, question=_QUESTION, choices=(), answer_type='int', gold=gold, accept={'graph_world': world}, slots=sl …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate` — L93**
```python
generate(index: int, seed: int, split: Split, mode: Mode, style: str='default') -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `graph.generate`, `_record`
- 반환 표현: `_record(original.verifier.accept['graph_world'], index, seed, split, mode, style)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`self_check` — L101**
```python
self_check(problem: GeneratedProblem) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `int`, `problem.actor.problem_id.split`, `world.items`, `matches_split`, `structure_hash`, `ValueError`, `_record`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.087 `fine_tuning/notebook_rl/generators/music.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/music.py) · SHA256 `0d175beaab4c4080f0041ea0df715518f9a8549376d53763e3b9b132f1e2cdfb`

모듈의 기존 설명: Program-authored notated music; independent theory checks, no teacher solutions.

**`_pitch` — L29**
```python
_pitch(diatonic: int, alteration: int) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`
- 반환 표현: `_STEPS[diatonic % 7] + ('#' * alteration if alteration >= 0 else 'b' * -alteration) + str(diatonic // 7)`

**`_midi` — L33**
```python
_midi(pitch: tuple[int, int] | list[int]) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `12 * (diatonic // 7 + 1) + _SEMITONES[diatonic % 7] + alter`

**`_signature` — L38**
```python
_signature(sharps: int) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `abs`
- 반환 표현: `{letter: 1 if sharps >= 0 else -1 for letter in letters[:abs(sharps)]}`

**`_select` — L43**
```python
_select(index: int, seed: int, split: Split, style: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `matches_split`, `random.Random`, `range`, `rng.choice`, `rng.randint`, `_signature`, `pitches.append`, `accidental.get`, `len`, `enumerate` 외 4개
- 반환 표현: `world`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`_manual_interval` — L71**
```python
_manual_interval(first, second) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `abs`, `_midi`, `str`
- 반환 표현: `quality + str(degree)`

**`_answers` — L84**
```python
_answers(world: dict, task: str, target: int) -> tuple[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_pitch`, `interval.Interval`, `note.Note`, `_manual_interval`, `tuple`, `_midi`, `qualities.get`, `ValueError`, `chord.Chord`, `key.KeySignature(world['sharps']).asKey('major').tonic.name.replace` 외 4개
- 반환 표현: `(interval.Interval(note.Note(names[0]), note.Note(names[1])).name, _manual_interval(*pitches[:2]))`; `(chord.Chord(names[:3]).quality, manual)`; `(key.KeySignature(world['sharps']).asKey('major').tonic.name.replace('-', 'b') + ' major', _MAJOR_KEYS[world['sharps']] + ' major')`; 외 2개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_xml` — L111**
```python
_xml(world: dict) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ET.Element`, `ET.SubElement`, `_signature`, `range`, `str`, `enumerate`, `active_accidentals.get`, `signature.get`, `ET.tostring`
- 반환 표현: `ET.tostring(root, encoding='utf-8', xml_declaration=True)`

**`_render` — L151**
```python
_render(world: dict, style: str) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `verovio.toolkit`, `toolkit.resetXmlIdSeed`, `toolkit.setOptions`, `ValueError`, `toolkit.loadData`, `_xml(world).decode`, `_xml`, `toolkit.getPageCount`, `cairosvg.svg2png`, `toolkit.renderToSVG(1).encode` 외 4개
- 반환 표현: `png_bytes(image, dpi=100)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate` — L167**
```python
generate(index: int, seed: int, split: Split, mode: Mode, style: str='default') -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `_select`, `len`, `_answers`, `range`, `', '.join`, `_pitch`, `zip`, `str` 외 5개
- 반환 표현: `build(GENERATOR_VERSION, FAMILY, index, split, mode, questions[task], (), 'short_text', answer, {'case_sensitive': True} if task == 'interval' else {}, slots, kinds, required, tuple((f' …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`self_check` — L198**
```python
self_check(p: GeneratedProblem) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `int`, `p.actor.problem_id.split`, `p.lineage.image_provenance.rsplit`, `generate`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.088 `fine_tuning/notebook_rl/generators/statics.py`

[현재 소스](../../fine_tuning/notebook_rl/generators/statics.py) · SHA256 `6396ef72f865eb8d9e9d585782985b9b8cab1bf66c014c3659da529974c3996f`

모듈의 기존 설명: Authored beam statics with independent exact solvers; not training-approved.

**`_select` — L28**
```python
_select(index: int, seed: int, split: Split, style: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `matches_split`, `random.Random`, `range`, `rng.randint`, `sorted`, `rng.sample`, `rng.choice`, `rng.getrandbits` 외 3개
- 반환 표현: `structure`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`_breaks` — L52**
```python
_breaks(structure: dict) -> list[Fraction]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Fraction`, `points.update`, `map`, `sorted`
- 반환 표현: `sorted(points)`

**`_equilibrium` — L60**
```python
_equilibrium(structure: dict) -> dict[str, Fraction]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Fraction`, `sum`, `set`, `_breaks`, `zip`, `max`, `min`, `candidates.add`, `abs`, `moment` 외 2개
- 반환 표현: `{'reaction': left, 'reaction_b': right, 'max_moment': maximum, 'max_location': min((x for x, value in magnitudes.items() if value == maximum))}`

**`_fraction` — L96**
```python
_fraction(value) -> Fraction
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `value.as_numer_denom`, `Fraction`, `int`
- 반환 표현: `Fraction(int(numerator), int(denominator))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_sympy` — L103**
```python
_sympy(structure: dict) -> dict[str, Fraction]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Beam`, `beam.apply_support`, `list`, `reactions.append`, `beam.apply_load`, `beam.solve_for_reaction_loads`, `beam.bending_moment`, `_breaks`, `zip`, `Rational` 외 13개
- 반환 표현: `{'reaction': _fraction(beam.reaction_loads[reactions[0]]), 'reaction_b': Fraction(0) if fixed else _fraction(beam.reaction_loads[at_b]), 'max_moment': maximum, 'max_location': min((p fo …`

**`_rounded` — L139**
```python
_rounded(value: Fraction) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `divmod`, `abs`
- 반환 표현: `f"{('-' if value < 0 and q else '')}{q // 10}.{q % 10}"`

**`_question` — L145**
```python
_question(task: str) -> tuple[str, str, list[str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `(question, unit, aliases)`

**`_slots` — L161**
```python
_slots(structure: dict) -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `enumerate`
- 반환 표현: `slots`

**`_render` — L175**
```python
_render(structure: dict, style: str) -> tuple[bytes, tuple[str, ...]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Figure`, `FigureCanvasAgg`, `figure.add_subplot`, `ax.set_facecolor`, `ax.plot`, `ax.add_patch`, `Polygon`, `ax.text`, `printed.append`, `enumerate` 외 15개
- 반환 표현: `(png_bytes(figure, dpi=110 if shifted else 100), tuple(printed))`

**`generate` — L240**
```python
generate(index: int, seed: int, split: Split, mode: Mode, style: str='default') -> GeneratedProblem
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `_select`, `_equilibrium`, `_sympy`, `len`, `_question`, `_rounded`, `_slots`, `_render`, `build` 외 2개
- 반환 표현: `build(source=GENERATOR_VERSION, family=FAMILY, index=index, split=split, mode=mode, question=question, choices=(), answer_type='unit_number', gold=gold, accept={'unit_aliases': aliases, …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`self_check` — L263**
```python
self_check(p: GeneratedProblem) -> None
```
- 기존 설명: Rebuild independent solutions, complete records, and PNG bytes from the identity.
- 직접 호출: `isinstance`, `ValueError`, `int`, `p.actor.problem_id.split`, `p.lineage.image_provenance.rsplit`, `generate`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.089 `fine_tuning/notebook_rl/gpu_gates.py`

[현재 소스](../../fine_tuning/notebook_rl/gpu_gates.py) · SHA256 `3204116767771e2928f6edc6e72c478accc0d9b49ca77d7975d21f7118711dd9`

모듈의 기존 설명: Live inventory and device numerical probes, not a complete G0--G13 approval.

**`_require` — L24**
```python
_require(condition: bool, message: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`measure_inventory` — L29**
```python
measure_inventory(bundle, protocol: dict) -> dict
```
- 기존 설명: G1 live evidence; never substitutes for checkpoint-file or GPU provenance.
- 직접 호출: `_require`, `_inventory`
- 반환 표현: `_inventory(bundle, protocol, expected_targets={'L': 252, 'V': 48})`

**`_inventory` — L35**
```python
_inventory(bundle, protocol: dict, *, expected_targets: dict) -> dict
```
- 기존 설명: Internal numeric-size seam permits tiny real Qwen CPU regression tests.
- 직접 호출: `bundle.model.get_base_model`, `dict`, `base.named_modules`, `named.items`, `type`, `_require`, `len`, `isinstance`, `set`, `all` 외 37개
- 반환 표현: `{'trainable_counts': counts, 'mergers': len(wrappers), 'master_dtype': 'float32', 'base_dtype': 'bfloat16', 'roots': {'vision': visual[0], 'text': text[0]}, 'targets': targets, 'install …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_sources` — L156**
```python
_sources() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_read`
- 반환 표현: `{name: _sha(_read(HERE / name)) for name in ('gpu_gates.py', 'gate_numerics.py', 'losses.py', 'policy.py', 'protocol_v1.yaml')}`

**`run_numerical_probe` — L161**
```python
run_numerical_probe(device: str, output: Path) -> dict
```
- 기존 설명: Write an actual-device, model-free check; it grants no training permission.
- 직접 호출: `Path`, `_require`, `output.is_absolute`, `output.exists`, `output.is_symlink`, `output.resolve().is_relative_to`, `output.resolve`, `_safe_root`, `_sources`, `time.monotonic` 외 6개
- 반환 표현: `result`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`main` — L193**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `run_numerical_probe`, `print`, `json.dumps`, `str`, `type`
- 반환 표현: `0`; `2`

### D18.090 `fine_tuning/notebook_rl/guards.py`

[현재 소스](../../fine_tuning/notebook_rl/guards.py) · SHA256 `e37f158845012e11d22e46d8965d09e422b0e59103dbb68345a857426fe5638a`

모듈의 기존 설명: One-shot Python audit checks, not an OS sandbox or a native-code boundary.

**`_deny_sdk_call` — L18**
```python
_deny_sdk_call(*args, **kwargs) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `PermissionError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `PermissionError`; 전파되는 callee 예외는 별도.

**`prepare_vllm_schema_dependency` — L22**
```python
prepare_vllm_schema_dependency() -> dict
```
- 기존 설명: Trusted startup only: vLLM needs OpenAI types, never an API client.
- 직접 호출: `RuntimeError`, `assert_no_teacher_imports`, `version`, `import_module`, `getattr`, `tuple`, `set`, `any`, `isinstance`, `callable` 외 1개
- 반환 표현: `{'dependency': 'openai', 'version': sdk.__version__, 'purpose': 'VLLM_SCHEMA_TYPES_ONLY', 'client_construction_and_requests_blocked': True, 'os_network_sandbox': False}`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`_teacher` — L53**
```python
_teacher(name: str) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `name.startswith`, `any`
- 반환 표현: `False`; `any((name == prefix or name.startswith(prefix + '.') for prefix in _TEACHERS))`

**`assert_no_teacher_imports` — L59**
```python
assert_no_teacher_imports() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `getattr`, `sys.modules.get`, `RuntimeError`, `_teacher`, `tuple`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**클래스 `_NoTeacherFinder` — L67**

기저: `object`.

**`_NoTeacherFinder.find_spec` — L68**
```python
_NoTeacherFinder.find_spec(self, fullname: str, path=None, target=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_teacher`, `PermissionError`
- 반환 표현: `None`
- 명시적 예외: `PermissionError`; 전파되는 callee 예외는 별도.

**`_paths` — L75**
```python
_paths(path: Path) -> tuple[Path, Path]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `os.path.abspath`, `absolute.resolve`
- 반환 표현: `(absolute, absolute.resolve())`

**`install_guards` — L80**
```python
install_guards(*, forbidden_roots: list[Path], vault_files: list[Path], allow_vault: bool, allow_benchmark: bool) -> None
```
- 기존 설명: Install before untrusted work; already-open handles/native I/O remain outside this check.
- 직접 호출: `RuntimeError`, `type`, `ValueError`, `tuple`, `_paths`, `assert_no_teacher_imports`, `sys.addaudithook`, `sys.meta_path.insert`, `_NoTeacherFinder`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `RuntimeError`, `ValueError`; 전파되는 callee 예외는 별도.

### D18.091 `fine_tuning/notebook_rl/input_preparation.py`

[현재 소스](../../fine_tuning/notebook_rl/input_preparation.py) · SHA256 `95b879f4b9b2914287849b68f054c616ae97384e5fb07e35c49ee7a524f743ca`

모듈의 기존 설명: Backend-neutral, pinned-processor input preparation.

**`digest` — L12**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':') …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()`

**`tensor_identity` — L17**
```python
tensor_identity(tensor: Any) -> dict[str, Any]
```
- 기존 설명: Return a stable CPU identity without changing the model input.
- 직접 호출: `tensor.detach().cpu().contiguous`, `tensor.detach().cpu`, `tensor.detach`, `list`, `str`, `hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest`, `hashlib.sha256`, `value.view(torch.uint8).numpy().tobytes`, `value.view(torch.uint8).numpy`, `value.view`
- 반환 표현: `{'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}`

**`prepare_inputs` — L27**
```python
prepare_inputs(processor: Any, model_config: Any, messages: list[dict[str, Any]], images: list[Any], **processor_kwargs: Any) -> dict[str, Any]
```
- 기존 설명: Apply the pinned processor once and validate its actual image tensors.
- 직접 호출: `processor.apply_chat_template`, `processor`, `ValueError`, `len`, `str`, `inputs['image_grid_thw'].is_floating_point`, `int`, `(inputs['input_ids'] == model_config.image_token_id).sum`, `inputs['image_grid_thw'].prod(-1).sum`, `inputs['image_grid_thw'].prod` 외 5개
- 반환 표현: `{'chat_prompt': text, 'inputs': inputs, 'input_tokens': int(inputs['input_ids'].shape[-1]), 'prompt_token_ids': inputs['input_ids'][0].tolist(), 'image_grid_thw': inputs['image_grid_thw …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare_protocol_inputs` — L54**
```python
prepare_protocol_inputs(processor: Any, model_config: Any, messages: list[dict[str, Any]], images: list[Any], image_cfg: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]
```
- 기존 설명: Use the same image path for the CPU reference and vLLM; never resize twice.
- 직접 호출: `image_cfg.get`, `prepare_inputs`, `ValueError`, `version`, `RuntimeError`, `process_vision_info`, `len`, `list`
- 반환 표현: `(prepare_inputs(processor, model_config, messages, images), images)`; `(item, processed)`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

### D18.092 `fine_tuning/notebook_rl/launch.py`

[현재 소스](../../fine_tuning/notebook_rl/launch.py) · SHA256 `199e23e7ee512d505e55c1f036fb423262b2395b0671297828d300219382bd0e`

모듈의 기존 설명: Run an approved notebook-RL job in a clean RunPod checkout under its watchdog.

**`read_json` — L38**
```python
read_json(path: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `_read`, `Path`, `type`, `ValueError`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`descriptor` — L47**
```python
descriptor(value: dict) -> Path
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `isinstance`, `Path(value['path']).is_absolute`, `Path`, `re.fullmatch`, `ValueError`
- 반환 표현: `Path(value['path'])`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`read_descriptor` — L56**
```python
read_descriptor(value: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `descriptor`, `_read`, `_sha`, `ValueError`, `read_json`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_job` — L67**
```python
validate_job(job: dict, protocol: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `type`, `isinstance`, `re.fullmatch`, `ValueError`, `utc`, `descriptor`, `Path(job[key]).is_absolute`, `Path`, `Path(job['resume']).is_absolute` 외 7개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`preflight` — L97**
```python
preflight(job: dict, protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_job`, `approval.validate_for_training`, `Path`
- 반환 표현: `approval.validate_for_training(Path(job['build_root']), Path(job['selection_path']), phase_a_report=Path(job['phase_a_report']), index=job['index'], source_registry=job['source_registry …`

**`checkpoint_binding` — L105**
```python
checkpoint_binding(validated: dict, protocol: dict, commit: str) -> dict
```
- 기존 설명: Shared with the pre-learning F0 monitor; runtime GPU evidence is not model identity.
- 직접 호출: `digest`, `validated['input_binding'].items`
- 반환 표현: `{'protocol_sha256': protocol['sha256'], 'prompts_sha256': protocol['prompts_sha256'], 'code_sha256': digest({'commit': commit, 'sources': validated['code_sha256']}), 'data_sha256': dige …`

**`runtime_check` — L114**
```python
runtime_check(job: dict, protocol: dict) -> dict
```
- 기존 설명: Read actual checkout, ancestor watchdog, lock, pinned weights and GPU identity.
- 직접 호출: `git`, `ValueError`, `scope.get`, `os.environ.get`, `import_module`, `int`, `psutil.Process().parents`, `psutil.Process`, `watcher.cmdline`, `argument` 외 38개
- 반환 표현: `{'device': actual, 'watchdog_pid': watcher_id, 'deadline_timestamp': deadline, 'free_memory_bytes': free, 'minimum_weights_and_kv_bytes': minimum, 'memory_fit_certified': False, 'weight …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_probes` — L214**
```python
load_probes(value: dict, scope: dict, image_root: Path, prompts: dict, stack: ExitStack) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read_descriptor`, `set`, `isinstance`, `len`, `ValueError`, `from_json`, `_bytes(row['actor']).decode`, `_bytes`, `_sha`, `declared.get` 외 10개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`execute` — L243**
```python
execute(job: dict, protocol: dict, validated: dict) -> dict
```
- 기존 설명: Production continuation; never pass synthetic approvals here in a real run.
- 직접 호출: `validated.get`, `ValueError`, `preflight`, `Path`, `checkpoint_binding`, `monitor.validate_f0_reference`, `descriptor`, `read_json`, `phase.get`, `_sha` 외 34개
- 반환 표현: `run_loop(bundle=bundle, optimizer=optimizer, scheduler=scheduler, engine=engine, preparation=preparation, verifier=verifier, actors=validated['actors'], structures=validated['structures …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L340**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_mutually_exclusive_group`, `mode.add_argument`, `parser.parse_args`, `read_json`, `load_protocol`, `preflight`, `print`, `json.dumps` 외 3개
- 반환 표현: `0 if result['status'] == 'VALIDATED_FOR_TRAINING' else 2`; `0 if result['status'] in ('STEP_LIMIT', 'EPOCH_COMPLETED') else 2`; `2`

### D18.093 `fine_tuning/notebook_rl/local_calibrate.py`

[현재 소스](../../fine_tuning/notebook_rl/local_calibrate.py) · SHA256 `90eb2a04ec4221c2e194196a2f17ca27f2601e273f6a9a41e91f7fdca29357dc`

모듈의 기존 설명: Measured CUDA/NF4 readiness; this command does not perform RL or claim improvement.

**`history_parent` — L25**
```python
history_parent(parent: Path, proof: Path, proof_sha256: str, actors: list, model_path: Path, problem_id: str) -> unannotated
```
- 기존 설명: Consume the pinned terminal auditor receipt, never a live/prefix capture.
- 직접 호출: `file_hash`, `ValueError`, `read`, `audit.get`, `Path(audit.get('capture', '')).resolve`, `Path`, `parent.resolve`, `type`, `launch.get`, `psutil.pid_exists` 외 28개
- 반환 표현: `(trees[ids.index(problem_id)], {'files': {str(p): hashes[str(p)] for p in paths} | {str(proof): proof_sha256}, 'protocol_sha256': protocol['sha256'], 'history_policy_version': trees[ids …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`history_request` — L104**
```python
history_request(actor, tree, images, prompts, engine, verifier) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `notebook.audit_episode`, `ValueError`, `for_actor`, `engine.prepare`, `RolloutRequest`, `user_message`, `turn1_text`, `len`, `notebook.solution_samples`, `engine.tokenizer.decode` 외 8개
- 반환 표현: `(request, audit)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`history_backward` — L145**
```python
history_backward(bundle, inputs: dict, completion_len: int) -> dict
```
- 기존 설명: One signed generated-action objective; no labels, rewards, or optimizer.
- 직접 호출: `set`, `any`, `bundle.params.values`, `bundle.model.parameters`, `ValueError`, `bundle.route.apply`, `id`, `bundle.model.named_parameters`, `bundle.model.train`, `torch.enable_grad` 외 12개
- 반환 표현: `{'objective': float(objective.detach()), 'gradients': summary, 'frozen_gradients_none': True, 'temporary_gradients_cleared': True}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`long_history_probe` — L176**
```python
long_history_probe(preparation: Path, model_path: Path, cache_check: Path, parent: Path, parent_audit: Path, parent_audit_sha256: str, problem_id: str, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `os.environ.get`, `os.getppid`, `ValueError`, `time.monotonic`, `output_path`, `Path`, `read`, `output.mkdir`, `os.getpid` 외 49개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`supervised_history` — L298**
```python
supervised_history(args) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `signal.signal`, `long_history_probe`, `getattr`, `name.replace`, `str`, `subprocess.Popen`, `dict`, `os.getpid`, `process.wait`, `terminate_group` 외 1개
- 반환 표현: `0`; `process.wait(timeout=MAX_SECONDS - 30)`; `124`

**`verify_original` — L329**
```python
verify_original(model_path: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(Path(__file__).parent / 'resources/baseline.json').read_text`, `Path`, `ValueError`, `path.is_file`, `path.stat`, `file_hash`
- 반환 표현: `checked`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`kernel_probe` — L342**
```python
kernel_probe() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.manual_seed`, `bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16, compress_statistics=Tr …`, `bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16, compress_statistics=Tr …`, `bnb.nn.Linear4bit`, `torch.randn`, `layer`, `y.float().square().mean().backward`, `y.float().square().mean`, `y.float().square`, `y.float` 외 11개
- 반환 표현: `{'status': 'KERNEL_FORWARD_BACKWARD_VERIFIED', 'shape': list(y.shape), 'input_grad_norm': float(x.grad.float().norm()), 'weight_grad': None, 'pretrained_model_test': False, 'optimizer_u …`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`policy_probe` — L361**
```python
policy_probe(model_path: Path, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_write`, `_bytes`, `verify_original`, `build_protocol`, `torch.manual_seed`, `build_quantized_policy`, `assert_quantized_base_unchanged`, `sum`, `p.numel`, `bundle.params.items` 외 1개
- 반환 표현: `{'status': 'PRETRAINED_NF4_POLICY_LOADED', 'optimizer_updates': 0, 'trainable_parameters': {k: sum((p.numel() for p in ps)) for k, ps in bundle.params.items()}, 'quantized_modules': len …`

**`main` — L379**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `any`, `getattr`, `parser.error`, `supervised_history`, `args.output.mkdir`, `time.monotonic`, `os.getpid` 외 17개
- 반환 표현: `supervised_history(args)`; `code`

### D18.094 `fine_tuning/notebook_rl/local_data.py`

[현재 소스](../../fine_tuning/notebook_rl/local_data.py) · SHA256 `db661aca202c631be6d7be12f789008f8aee744146933f94c4c07eb3c0a9415e`

모듈의 기존 설명: Rebind unchanged reviewed graph inputs for explicitly authorized local NF4 work.

**`_write_json` — L29**
```python
_write_json(path: Path, value: dict | list) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_write`, `_bytes`, `evidence._descriptor`
- 반환 표현: `evidence._descriptor(path)`

**`project_observation_actors` — L34**
```python
project_observation_actors(actors: list[ActorItem], protocol: dict) -> list[ActorItem]
```
- 기존 설명: Change elicitation only; private answers and reference slots are not inputs.
- 직접 호출: `protocol.get`, `ValueError`, `len`, `to_json`, `replace`
- 반환 표현: `[replace(actor, mode='obs') for actor in actors]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_split_summary` — L45**
```python
_split_summary(root: Path, selected: list[str]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read_actors`, `read_lineage`, `set`, `any`, `ValueError`, `train.keys`, `rows.items`, `lineage.values`, `len`, `sorted`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare_local_data` — L74**
```python
prepare_local_data(previous_request: Path, selection: Path, protocol: dict, out: Path) -> dict
```
- 기존 설명: Write new proof bindings, then run the existing isolated D1–D8 validator.
- 직접 호출: `protocol.get`, `type`, `authority.get`, `authority['reference'].strip`, `ValueError`, `evidence._hash`, `json.loads`, `json.dumps`, `protocol.items`, `Path(previous_request).absolute` 외 37개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `FileExistsError`; 전파되는 callee 예외는 별도.

### D18.095 `fine_tuning/notebook_rl/local_eval.py`

[현재 소스](../../fine_tuning/notebook_rl/local_eval.py) · SHA256 `189a86da16990dfb95d4c9b0fb333550441de3c7ff158e96517cc810f40ee75d`

모듈의 기존 설명: Fixed-ten NF4 circuit diagnosis; no training, teacher, or official-score claim.

**`wire` — L62**
```python
wire(value) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ': …`, `json.dumps`
- 반환 표현: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()`

**`unique` — L67**
```python
unique(pairs) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`decode` — L76**
```python
decode(raw) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`
- 반환 표현: `json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)`

**`digest` — L82**
```python
digest(value) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(wire(value)).hexdigest`, `hashlib.sha256`, `wire`
- 반환 표현: `hashlib.sha256(wire(value)).hexdigest()`

**`read` — L86**
```python
read(path: Path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `decode`, `_read`
- 반환 표현: `decode(_read(path))`

**`save` — L91**
```python
save(path: Path, value) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_write`, `wire`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`source_hashes` — L96**
```python
source_hashes() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(package.__file__).resolve`, `Path`, `root.glob`, `Path(__file__).resolve`, `approval.code_hashes`, `str`, `hashlib.sha256(p.read_bytes()).hexdigest`, `hashlib.sha256`, `p.read_bytes`, `sorted` 외 1개
- 반환 표현: `{'approval': approval.code_hashes(), 'evaluation': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}}`

**`protocol` — L105**
```python
protocol() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{**EVALUATION_PROTOCOL, 'protocol_id': VERSION, 'input_protocol_id': EVALUATION_PROTOCOL['protocol_id'], 'recovery_protocol': VERSION, 'max_attempts': 3, 'context_limit': CONTEXT_LIMIT, …`

**`_response` — L118**
```python
_response(row: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{k: row[k] for k in ('problem_id', 'text', 'finish_reason')}`

**`_first_attempt_lines` — L123**
```python
_first_attempt_lines(text: str) -> list[dict]
```
- 기존 설명: Lossless Unicode-character offsets; never normalize or strip line endings.
- 직접 호출: `enumerate`, `text.splitlines`, `spans.append`, `len`
- 반환 표현: `spans`

**`_answer` — L132**
```python
_answer(row: dict) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `extract_response`
- 반환 표현: `extract_response(row['text'], 'qa') if row['finish_reason'] in ('stop', 'eos') else None`; `None`

**`note_feedback` — L140**
```python
note_feedback(history: list[dict], statuses: list[str]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `ValueError`, `_first_attempt_lines`, `enumerate`, `zip`, `wire(transcript).decode().replace('<', '\\u003c').replace`, `wire(transcript).decode().replace`, `wire(transcript).decode`, `wire`, `NOTE.format`
- 반환 표현: `NOTE.format(transcript=raw)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`score_note` — L152**
```python
score_note(note: dict, history: list[dict], statuses: list[str], structure_target: str) -> dict
```
- 기존 설명: CPU-only per-attempt citations/answers and registered circuit; no cause claim.
- 직접 호출: `len`, `any`, `row.get`, `enumerate`, `ValueError`, `digest`, `decode`, `type`, `set`, `obj['lesson'].strip` 외 14개
- 반환 표현: `checks | {'verified': False}`; `checks | {'verified': all((checks[k] for k in ('format_valid', 'history_verified', 'previous_answer_verified', 'corrected_answer_verified', 'corrected_structure_verified')))}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_call` — L209**
```python
_validate_call(row: dict, pid: str, stage: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `row.get`, `seed`, `ValueError`, `any`, `digest`, `hashlib.sha256(row['text'].encode()).hexdigest`, `hashlib.sha256`, `row['text'].encode`, `len` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_note_messages` — L244**
```python
_note_messages(actor, feedback: str | None) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `feedback.strip`, `ValueError`, `external_messages`
- 반환 표현: `messages`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_audit_frontiers` — L253**
```python
_audit_frontiers(cohort: Path, journals: list, responses: list, actors: list) -> tuple[dict, dict]
```
- 기존 설명: Derive histories from every freshly regraded full frontier, never a supplied first answer.
- 직접 호출: `type`, `len`, `ValueError`, `enumerate`, `any`, `r.get`, `zip`, `_validate_call`, `row.get`, `list` 외 8개
- 반환 표현: `(histories, outcomes)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`worker` — L306**
```python
worker(payload: dict) -> dict
```
- 기존 설명: Only this fresh process reads target bytes; responses contain bounded booleans.
- 직접 호출: `type`, `set`, `ValueError`, `Path`, `install_guards`, `Path.home`, `score_external_responses`, `_actor_inputs`, `_audit_frontiers`, `len` 외 11개
- 반환 표현: `{'status': 'INCOMPLETE', 'n_received': scored['n_received'], 'rows': [], **EVIDENCE_LIMITS}`; `{'status': 'COMPLETE', 'n_correct': scored['n_correct'], 'rows': rows, 'responses_sha256': scored['responses_sha256'], **EVIDENCE_LIMITS}`; `{'status': 'COMPLETE', 'problem_id': pid, 'frontiers_sha256': digest(request['frontiers']), 'checks': score_note(note, history, statuses, target[0]['target']), **EVIDENCE_LIMITS}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`trusted` — L359**
```python
trusted(cohort: Path, responses: list[dict], note: dict | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`, `wire`, `len`, `ValueError`, `dict`, `Path(fine_tuning.__file__).resolve`, `Path`, `environment.get`, `subprocess.run`, `Path(__file__).resolve` 외 4개
- 반환 표현: `value | {'scorer_request_bytes': len(raw), 'scorer_response_bytes': len(result.stdout)}`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`frontier` — L380**
```python
frontier(actors, latest: dict) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `ValueError`, `_response`
- 반환 표현: `[_response(latest[a.problem_id]) for a in actors]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`continuation` — L386**
```python
continuation(row: dict, feedback: str | None, tokenizer) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `SimpleNamespace`, `continuation_ids`
- 반환 표현: `continuation_ids(history, feedback, tokenizer)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`seed` — L395**
```python
seed(pid: str, stage: str) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int.from_bytes`, `hashlib.sha256(f'3407:{VERSION}:{pid}:{stage}'.encode()).digest`, `hashlib.sha256`, `f'3407:{VERSION}:{pid}:{stage}'.encode`
- 반환 표현: `int.from_bytes(hashlib.sha256(f'3407:{VERSION}:{pid}:{stage}'.encode()).digest()[:8], 'big') % 2 ** 31`

**`generate` — L400**
```python
generate(actor, images, engine, stage: str, output: Path, *, parent: dict | None=None, feedback: str | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `continuation`, `_note_messages`, `external_messages`, `RolloutRequest`, `seed`, `engine.prepare`, `time.monotonic`, `_sample_request`, `digest`, `hashlib.sha256(completion.text.encode()).hexdigest` 외 9개
- 반환 표현: `row`

**`diagnose` — L435**
```python
diagnose(actors, image_loader, engine, cohort: Path, output: Path, *, deadline: float) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `call`, `journals.append`, `trusted`, `frontier`, `save`, `ValueError`, `dict`, `statuses.items`, `histories[pid].append`, `outcomes[actor.problem_id].append` 외 11개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`evaluate` — L526**
```python
evaluate(capture: Path, external_preparation: Path, output: Path, *, condition: str, reload_result: Path, minimal_capture: Path, minimal_reload: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_evaluate`
- 반환 표현: `_evaluate(capture, external_preparation, output, condition=condition, reload_result=reload_result, minimal_capture=minimal_capture, minimal_reload=minimal_reload)`

**`evaluate_ladder` — L532**
```python
evaluate_ladder(capture: Path, cohort: Path, data_gate: Path, output: Path, *, condition: str, arm: str, reload_result: Path, minimal_capture: Path | None=None, minimal_reload: Path | None=None, with_recovery: bool=False, evaluation_prompt_policy: str | None=None) -> dict
```
- 기존 설명: Native evaluation after explicit data/learning/reload gates.
- 직접 호출: `_evaluate`
- 반환 표현: `_evaluate(capture, data_gate, output, condition=condition, arm=arm, ladder_cohort=cohort, reload_result=reload_result, minimal_capture=minimal_capture, minimal_reload=minimal_reload, wi …`

**`_evaluate` — L545**
```python
_evaluate(capture: Path, external_preparation: Path, output: Path, *, condition: str, reload_result: Path, minimal_capture: Path | None, minimal_reload: Path | None, ladder_cohort: Path | None=None, arm: str | None=None, with_recovery: bool=False, evaluation_prompt_policy: str | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `time.monotonic`, `read`, `restore_protocol`, `launch.get('execution_contract', {}).get`, `launch.get`, `checkpoint.digest`, `execution_max_seconds`, `execution_admission` 외 105개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `TimeoutError`, `re-raise`; 전파되는 callee 예외는 별도.

**`_compare_native_recovery` — L924**
```python
_compare_native_recovery(before: dict, after: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `before.get`, `after.get`, `ValueError`, `before['conditions'].get('protocol', {}).get`, `before['conditions'].get`, `type`, `before['conditions']['protocol'].get`, `any`, `required_conditions.items`, `result.get` 외 6개
- 반환 표현: `{**paired, 'status': 'COMPLETE_PAIRED_NATIVE_LADDER_RECOVERY', 'n_paired': 10, 'before_sha256': digest(before), 'after_sha256': digest(after), 'execution_conditions_verified': True, 'ge …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`compare` — L973**
```python
compare(before: dict, after: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_compare_native_recovery`, `before.get`, `after.get`, `len`, `set`, `any`, `r.get`, `ValueError`, `old.values`, `new.values` 외 2개
- 반환 표현: `_compare_native_recovery(before, after)`; `{'status': 'COMPLETE_PAIRED_DIAGNOSTIC', 'n_paired': 10, 'before_sha256': digest(before), 'after_sha256': digest(after), 'first_attempt_net_correct': sum((v == 'CORRECT' for v in new.va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L1002**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sys.stdin.buffer.read`, `len`, `wire`, `worker`, `decode`, `sys.stdout.buffer.write`, `argparse.ArgumentParser`, `parser.add_subparsers`, `sub.add_parser`, `run.add_argument` 외 22개
- 반환 표현: `2`; `0`; `0 if result['status'].startswith('COMPLETE') or result['status'] == 'CPU_NATIVE_LADDER_RECOVERY_REPLAYED' else 1`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.096 `fine_tuning/notebook_rl/local_external.py`

[현재 소스](../../fine_tuning/notebook_rl/local_external.py) · SHA256 `5206299701037c574d6a0be5edd2010f30c3d5c2552c4e83eef2f952fbef9181`

모듈의 기존 설명: Frozen circuit evaluation inputs and a separate trusted, program-only scorer.

**`_json` — L35**
```python
_json(path: Path, expected_sha256: str | None=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_read`, `_sha`, `ValueError`, `json.loads`
- 반환 표현: `json.loads(raw, object_pairs_hook=_unique)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_rows` — L42**
```python
_rows(path: Path, expected_sha256: str) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_read`, `_sha`, `ValueError`, `json.loads`, `raw.splitlines`, `any`, `type`
- 반환 표현: `rows`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_descriptor` — L52**
```python
_descriptor(path: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`, `path.absolute`, `_sha`, `_read`
- 반환 표현: `{'path': str(path.absolute()), 'file_sha256': _sha(_read(path))}`

**`_actor_inputs` — L56**
```python
_actor_inputs(cohort: Path, expected_manifest_sha256: str) -> tuple[dict, list[ActorItem], list[dict]]
```
- 기존 설명: Only public inputs, split metadata, manifest and image bytes are opened.
- 직접 호출: `_safe_root`, `Path(cohort).absolute`, `Path`, `_json`, `manifest.get`, `ValueError`, `_rows`, `len`, `set`, `any` 외 10개
- 반환 표현: `(manifest, actors, selected)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_external_actors` — L99**
```python
load_external_actors(cohort: Path, *, expected_manifest_sha256: str=FROZEN_MANIFEST_SHA256) -> list[ActorItem]
```
- 기존 설명: Gold-free actor boundary; options remain in the original circuit image.
- 직접 호출: `_actor_inputs`
- 반환 표현: `_actor_inputs(cohort, expected_manifest_sha256)[1]`

**`external_messages` — L104**
```python
external_messages(actor: ActorItem) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_actor`, `len`, `ValueError`, `user_message`
- 반환 표현: `[user_message(actor.question, len(actor.image_paths))]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare_external_data` — L111**
```python
prepare_external_data(cohort: Path, train_actors: list[ActorItem], train_structures: dict[str, str], out: Path, *, source_registry: Path, historical_approval: Path, expected_manifest_sha256: str=FROZEN_MANIFEST_SHA256) -> dict
```
- 기존 설명: Prepare private CPU metadata; no new scan, training, model load or evaluation.
- 직접 호출: `Path(cohort).absolute`, `Path`, `_actor_inputs`, `len`, `set`, `train_structures.values`, `ValueError`, `validate_actor`, `_json`, `registry.get` 외 22개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `FileExistsError`; 전파되는 callee 예외는 별도.

**`score_external_responses` — L175**
```python
score_external_responses(cohort: Path, responses: list[dict], *, expected_manifest_sha256: str=FROZEN_MANIFEST_SHA256) -> dict
```
- 기존 설명: Trusted scorer: open gold only after complete actual-response coverage checks.
- 직접 호출: `_actor_inputs`, `type`, `any`, `set`, `ValueError`, `len`, `_sha`, `_bytes`, `sorted`, `Path(cohort).absolute` 외 7개
- 반환 표현: `base | {'missing_ids': sorted(set(ids) - set(by_id))}`; `base | {'status': 'COMPLETE', 'gold_opened': True, 'n_correct': correct, 'accuracy': correct / 10, 'rows': scored, 'responses_sha256': _sha(_bytes(responses))}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`compare_external_scores` — L216**
```python
compare_external_scores(before: dict, after: dict) -> dict
```
- 기존 설명: Paired arithmetic only; the caller must attest actual identical run conditions.
- 직접 호출: `before.get`, `after.get`, `any`, `ValueError`, `set`, `len`, `type`, `old.values`, `new.values`, `sum`
- 반환 표현: `{'n_paired': 10, 'before_correct': sum(old.values()), 'after_correct': sum(new.values()), 'wrong_to_correct': sum((not old[k] and new[k] for k in old)), 'correct_to_wrong': sum((old[k]  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.097 `fine_tuning/notebook_rl/local_ladder.py`

[현재 소스](../../fine_tuning/notebook_rl/local_ladder.py) · SHA256 `6e5b809ddaeae3c8809a286171d79998ac532df9f7d41bfd3039282ac906f4e7`

모듈의 기존 설명: Prepare ten native sealed ladders on CPU; grants no evaluation/training authority.

**`_path` — L22**
```python
_path(value: str | Path) -> Path
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `path.is_absolute`, `ValueError`, `_safe_root`
- 반환 표현: `path`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_descriptor` — L30**
```python
_descriptor(path: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_path`, `str`, `_sha`, `_read`
- 반환 표현: `{'path': str(path), 'file_sha256': _sha(_read(path))}`

**`_json` — L35**
```python
_json(path: Path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `_read`, `_path`
- 반환 표현: `json.loads(_read(_path(path)), object_pairs_hook=_unique)`

**`_bound` — L39**
```python
_bound(descriptor: dict, bindings: dict[str, str]) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `ValueError`, `_path`, `_descriptor`, `str`, `_json`
- 반환 표현: `_json(path)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_sources` — L50**
```python
_sources() -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `paths.append`, `Path(module.__file__).absolute`, `Path`, `paths.extend`, `Path(__file__).absolute`, `str`, `_descriptor`, `sorted`
- 반환 표현: `{str(path): _descriptor(path)['file_sha256'] for path in sorted(paths)}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_graph_nodes` — L62**
```python
_graph_nodes(world: object) -> int
```
- 기존 설명: Count the actual native world, rejecting extensions to its bounded contract.
- 직접 호출: `isinstance`, `ValueError`, `graph._validate_world`, `len`, `set`, `world['trans'].values`, `row.values`, `world['children'].values`
- 반환 표현: `len(set(world['nodes']))`; `len(set(world['trans']) | {v for row in world['trans'].values() for v in row.values()})`; `len({world['root']} | set(world['children']) | {v for row in world['children'].values() for v in row.values()})`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_training` — L78**
```python
_training(preparation: Path) -> tuple[dict, list[ActorItem], dict, dict[str, str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_bound`, `_descriptor`, `prep.get`, `ValueError`, `_safe_root`, `Path`, `Path(prep['build_root']).is_absolute`, `request.get`, `str`, `validated.get` 외 26개
- 반환 표현: `(prep, selected, worlds, bindings)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_preserved` — L155**
```python
_preserved(sources: dict[str, str], bindings: dict[str, str]) -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sources`, `any`, `_descriptor`, `Path`, `bindings.items`, `ValueError`
- 반환 표현: `current`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare_ladder_data` — L163**
```python
prepare_ladder_data(preparation: Path, out: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_path`, `Path(preparation).absolute`, `Path`, `Path(out).absolute`, `any`, `path.is_symlink`, `ValueError`, `out.exists`, `FileExistsError`, `_sources` 외 22개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `FileExistsError`; 전파되는 callee 예외는 별도.

**`load_ladder_actors` — L234**
```python
load_ladder_actors(cohort: Path) -> list[ActorItem]
```
- 기존 설명: Public vault loader: never opens verifier/gold or lineage payloads.
- 직접 호출: `_safe_root`, `_json`, `manifest.get`, `_descriptor`, `ValueError`, `read_actors`, `len`, `load_images`, `image.close`
- 반환 표현: `actors`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L251**
```python
main(argv: list[str] | None=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `prepare_ladder_data`, `print`, `json.dumps`
- 반환 표현: `0`

### D18.098 `fine_tuning/notebook_rl/local_ladder_gate.py`

[현재 소스](../../fine_tuning/notebook_rl/local_ladder_gate.py) · SHA256 `49c80f2065e3812ff55300f9b82017eaf466b08e91486947e3ef2959854e44d2`

모듈의 기존 설명: Trusted CPU data audit and public-only binding for native ladder diagnosis.

**`_sources` — L23**
```python
_sources() -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(__file__).resolve`, `Path`, `ladder._sources`, `str`, `code_hashes().items`, `code_hashes`
- 반환 표현: `ladder._sources() | {str(root / name): sha for name, sha in code_hashes().items()}`

**`_validate_protocol` — L29**
```python
_validate_protocol(protocol: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `any`, `protocol.get`, `re.fullmatch`, `ValueError`, `local_ladder_generation.circuit_evaluation_prompts`, `module.protocol`, `_bytes`
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_public_inputs` — L47**
```python
_public_inputs(cohort: Path, preparation: Path) -> tuple[list, dict[str, str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ladder.load_ladder_actors`, `ladder._json`, `ladder._descriptor`, `Path`, `ValueError`, `files.append`, `files.extend`, `str`
- 반환 표현: `(actors, {str(p): ladder._descriptor(p)['file_sha256'] for p in files})`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_review` — L66**
```python
_review(path: Path, cohort: Path, actors: list) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ladder._json`, `list`, `review.get`, `type`, `ladder._descriptor`, `ValueError`
- 반환 표현: `review`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`build_gate` — L81**
```python
build_gate(cohort: Path, training_preparation: Path, review: Path, output: Path, protocol: dict) -> dict
```
- 기존 설명: Run only in a trusted CPU process: private reference data is read here.
- 직접 호출: `_safe_root`, `map`, `output.exists`, `output.is_relative_to`, `ValueError`, `_validate_protocol`, `_sources`, `ladder._json`, `_public_inputs`, `ladder._training` 외 29개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_gate` — L155**
```python
validate_gate(cohort: Path, training_preparation: Path, receipt: Path, protocol: dict) -> dict
```
- 기존 설명: Public binding check: does not open gold/lineage or load a model.
- 직접 호출: `_safe_root`, `_validate_protocol`, `map`, `ladder._json`, `_public_inputs`, `result.get`, `ladder._descriptor`, `_sha`, `_bytes`, `dict.fromkeys` 외 4개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.099 `fine_tuning/notebook_rl/local_ladder_generation.py`

[현재 소스](../../fine_tuning/notebook_rl/local_ladder_generation.py) · SHA256 `2c988bc28add0e72b8147597dcccebaeccc1cbd52b36edbca9d80784c1281f62`

모듈의 기존 설명: Native first-attempt ladder inference core; callers own all execution/data gates.

**`circuit_evaluation_prompts` — L33**
```python
circuit_evaluation_prompts(policy: str=CIRCUIT_PROMPT_POLICY) -> dict
```
- 기존 설명: Explicit evaluation-only templates; graph training trace prompts stay separate.
- 직접 호출: `prompts.load_prompts`, `Path(__file__).with_name`, `Path`, `ValueError`
- 반환 표현: `templates | {'evaluation_prompt_policy': policy}`; `templates | {'evaluation_prompt_policy': policy, 'solution_trace_contract': circuit_trace.VERSION, 'source_observation_instruction': {circuit.GENERATOR_VERSION: circuit.OBSERVATION_INST …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_digest` — L47**
```python
_digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_bytes`
- 반환 표현: `_sha(_bytes(value))`

**`_sha256` — L51**
```python
_sha256(value: Any) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `re.fullmatch`
- 반환 표현: `type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None`

**`protocol` — L55**
```python
protocol(prompt_templates: dict, local_protocol_sha256: str, *, context_limit: int=CONTEXT_LIMIT) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `_sha256`, `ValueError`, `circuit_evaluation_prompts`, `_digest`, `prompts.circuit_solution_prompt_contract`, `policy.update`
- 반환 표현: `{'protocol_id': VERSION, 'local_protocol_sha256': local_protocol_sha256, 'prompt_templates_sha256': _digest(prompt_templates), 'n': 1, 'max_attempts': 1, 'max_new_tokens': OUTPUT_TOKENS …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_seed` — L83**
```python
_seed(problem_id: str) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`
- 반환 표현: `seeds(problem_id, 0, 1, VERSION, 1, master_seed=3407)[0]`

**`_input` — L87**
```python
_input(actor, request: RolloutRequest, engine: Any, *, context_limit: int) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `engine.prepare`, `prepared.pixels.get`, `type`, `set`, `list`, `len`, `grid.is_floating_point`, `any`, `bool`, `(grid > 0).all` 외 7개
- 반환 표현: `{'actor': json.loads(to_json(actor)), 'messages': request.messages, 'prompt_token_ids': list(prepared.token_ids), 'image_key': prepared.image_key, 'input_tokens': len(prepared.token_ids …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_ladder` — L109**
```python
generate_ladder(cohort: Path, output: Path, engine: Any, prompt_templates: dict, *, expected_preparation_sha256: str, local_protocol_sha256: str, deadline: float | None=None, context_limit: int=CONTEXT_LIMIT) -> dict
```
- 기존 설명: Record exactly ten native first attempts from an already gated live engine.
- 직접 호출: `time.monotonic`, `type`, `math.isfinite`, `ValueError`, `check_deadline`, `_safe_root`, `Path(output).absolute`, `Path`, `any`, `path.is_symlink` 외 29개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`_validate_row` — L224**
```python
_validate_row(row: dict, *, context_limit: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `row.get`, `_digest`, `_seed`, `any`, `len`, `math.isfinite`, `inp.get`, `tokens.count` 외 1개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_record` — L255**
```python
validate_record(result: dict) -> None
```
- 기존 설명: Validate one complete literal first-attempt record without inventing a pair.
- 직접 호출: `type`, `result.get`, `ValueError`, `circuit_evaluation_prompts`, `protocol`, `contract.get`, `_bytes`, `_sha256`, `_digest`, `len` 외 11개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_paired_records` — L292**
```python
validate_paired_records(left: dict, right: dict) -> dict
```
- 기존 설명: Check recorded F0/FT conditions only; records cannot certify model provenance.
- 직접 호출: `validate_record`, `any`, `ValueError`, `zip`
- 반환 표현: `{'status': 'RECORDED_PAIRED_CONDITIONS_MATCH', 'n_pairs': 10, 'model_generation_provenance_verified': False, 'accuracy': None}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.100 `fine_tuning/notebook_rl/local_ladder_recovery.py`

[현재 소스](../../fine_tuning/notebook_rl/local_ladder_recovery.py) · SHA256 `3528b2f07c5093a6a049db9291de5559d530bc591a909e4395dac01b63dfd4bd`

모듈의 기존 설명: Native n=1 recovery and strict whole-history notes; no training or model loading.

**`_reference` — L36**
```python
_reference(templates: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `templates.get`
- 반환 표현: `templates.get('evaluation_prompt_policy') == generation.CIRCUIT_REFERENCE_PROMPT_POLICY`

**`protocol` — L40**
```python
protocol(prompt_templates: dict, local_protocol_sha256: str, *, context_limit: int=generation.CONTEXT_LIMIT) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `generation.protocol`, `_reference`, `reference.evaluation_generation_contract`, `generation._digest`, `sorted`
- 반환 표현: `{**first, 'protocol_id': VERSION, 'max_attempts': 3, 'first_attempt_protocol_id': generation.VERSION, 'seed_rule': 'T1 unchanged; retries/NOTE rollout.seeds(policy_version=0,branch=prot …`

**`_seed` — L70**
```python
_seed(pid: str, stage: str) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`
- 반환 표현: `seeds(pid, 0, 2 if stage == 'T2' else 3, VERSION + ':' + stage, 1, master_seed=3407)[0]`

**`_wire` — L74**
```python
_wire(value) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `to_json`
- 반환 표현: `json.loads(to_json(value))`

**`_group` — L78**
```python
_group(sample: Sample) -> Group
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Group`, `_hash`
- 반환 표현: `Group(sample.group_id, sample.kind, sample.problem_id, _hash(sample.prompt_token_ids), [sample], False, _SINGLE)`

**`_audit` — L84**
```python
_audit(tree, actor, verifier, templates: dict | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_reference`, `synthesis.audit_episode`, `notebook.audit_episode`
- 반환 표현: `synthesis.audit_episode(tree, actor, verifier, contract=synthesis.BOUNDED_VERSION, evaluation_only=True)`; `notebook.audit_episode(tree, actor, verifier, routing_rule=notebook.STRICT_ROUTE_RULE, evaluation_only=True)`

**`_note_messages` — L92**
```python
_note_messages(actor, templates, tree, audit, n_images) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `prompts.for_actor(actor, templates)['observation_instruction'].get`, `prompts.for_actor`, `type`, `observation.strip`, `ValueError`, `_reference`, `notebook.solution_samples`, `reference.validate_audit_coverage`, `reference.public_inventory`, `reference.evaluation_generation_text` 외 3개
- 반환 표현: `[prompts.user_message(body, n_images)]`; `[prompts.user_message(prompts.p0_body(actor.question, actor.choices) + '\n\n' + body, n_images)]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_grade_note` — L107**
```python
_grade_note(row, tree, audit, verifier, actor, templates) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_reference`, `reference.grade_note`, `notebook.grade_note`
- 반환 표현: `reference.grade_note(row['text'], row['finish_reason'], tree, audit, verifier, evaluation_only=True, actor=actor, contract=synthesis.BOUNDED_VERSION)`; `notebook.grade_note(row['text'], row['finish_reason'], tree, audit, verifier, evaluation_only=True, actor=actor)`

**`_check_verifier` — L116**
```python
_check_verifier(verifier, preparation) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_verifier` — L122**
```python
_verifier(root) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `VerifierService`, `Path.home`
- 반환 표현: `VerifierService(root / 'records', forbidden_roots=[Path.home() / 'mmdl-data/evaluation'], timeout=60)`

**`_sample` — L126**
```python
_sample(row, kind, turn, parent, verifier, actor) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Sample`, `list`, `verifier.grade`
- 반환 표현: `sample`

**`_validate_call` — L135**
```python
_validate_call(row, pid, version, first, parent=None, *, context_limit, note_tokens=2048) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `type`, `_seed`, `generation._digest`, `row.items`, `len`, `any`, `math.isfinite`, `ValueError`, `inp.get` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_metrics` — L177**
```python
_metrics(trees, rows, *, reference_policy=False) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `from_json`, `json.dumps`, `notebook.solution_samples`, `entry.get`, `type`, `checked.get`, `ValueError`, `any`, `strict.get`, `synthesis.successful` 외 8개
- 반환 표현: `{'first_statuses': first, 'final_statuses': final, 'first_attempt_accuracy': sum((status == 'CORRECT' for status in first.values())) / 10, 'final_accuracy': accuracy, 'recovery_accuracy …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`diagnose` — L221**
```python
diagnose(cohort: Path, output: Path, engine: Any, prompt_templates: dict, *, expected_preparation_sha256: str, local_protocol_sha256: str, deadline: float, context_limit: int=generation.CONTEXT_LIMIT) -> dict
```
- 기존 설명: All T1 calls precede n=1 retries, followed by one strict note per recovery.
- 직접 호출: `type`, `math.isfinite`, `ValueError`, `_safe_root`, `Path(output).absolute`, `Path`, `output.exists`, `output.is_relative_to`, `any`, `p.is_symlink` 외 42개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_complete` — L369**
```python
_validate_complete(record) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `record.get`, `generation._digest`, `record.items`, `ValueError`, `any`, `protocol`, `record['protocol'].get`, `_bytes`, `generation.validate_record` 외 25개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`replay` — L455**
```python
replay(cohort: Path, record: dict, *, expected_preparation_sha256: str) -> dict
```
- 기존 설명: Fresh private CPU regrading, without a model or tokenizer load.
- 직접 호출: `_validate_complete`, `_safe_root`, `generation._digest`, `_descriptor`, `load_ladder_actors`, `code_hashes`, `ValueError`, `_verifier`, `_check_verifier`, `_json` 외 8개
- 반환 표현: `{'status': 'CPU_NATIVE_LADDER_RECOVERY_REPLAYED', 'record_sha256': before, 'verifier_process': metadata, **{key: record[key] for key in _METRICS}, **({'metric_identity': record['protoco …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_paired_records` — L492**
```python
validate_paired_records(left: dict, right: dict) -> dict
```
- 기존 설명: Recorded conditions and arithmetic, not model provenance or learned causality.
- 직접 호출: `_validate_complete`, `type`, `set`, `generation._sha256`, `ValueError`, `replay`, `Path`, `generation.validate_paired_records`, `any`
- 반환 표현: `{'status': 'RECORDED_PAIRED_CONDITIONS_MATCH', 'n_pairs': 10, 'first_gains': gains, 'first_losses': losses, 'first_attempt_accuracy_delta': right['first_attempt_accuracy'] - left['first …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.101 `fine_tuning/notebook_rl/local_ladder_score.py`

[현재 소스](../../fine_tuning/notebook_rl/local_ladder_score.py) · SHA256 `0dabd8e8727f4dd05d384c670a44937297564a38975802876e91a7be72e886d2`

모듈의 기존 설명: CPU replay of native ladder answers; does not authorize or prove inference.

**`_code` — L21**
```python
_code() -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(__file__).resolve`, `Path`, `_sha`, `_read`
- 반환 표현: `{name: _sha(_read(root / name)) for name in names}`

**`score_responses` — L29**
```python
score_responses(cohort: Path, responses: list[dict], *, expected_preparation_sha256: str) -> dict
```
- 기존 설명: Read only public inputs in this process; a fresh verifier owns private gold.
- 직접 호출: `_safe_root`, `_descriptor`, `type`, `re.fullmatch`, `ValueError`, `_json`, `_code`, `load_ladder_actors`, `preparation.get`, `any` 외 15개
- 반환 표현: `{'status': 'CPU_RESPONSES_SCORED' if complete else 'INCOMPLETE_UNVERIFIABLE_RESPONSES', 'scoring_version': VERSION, 'preparation': identity, 'responses_sha256': response_sha, 'source_sh …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L80**
```python
main(argv: list[str] | None=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `_safe_root`, `_path`, `args.output.absolute`, `output.is_relative_to`, `output.exists`, `any`, `p.is_symlink` 외 11개
- 반환 표현: `0 if result['status'] == 'CPU_RESPONSES_SCORED' else 2`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.102 `fine_tuning/notebook_rl/local_profile.py`

[현재 소스](../../fine_tuning/notebook_rl/local_profile.py) · SHA256 `89888febe8dd22916aca82d5e63d4acdd6101cc01e942a5317ff49c6b816b3cf`

모듈의 기존 설명: Explicit 8GB local experiment; no change to the historical BF16 protocol.

**`build_protocol` — L58**
```python
build_protocol(*, verified_note_atoms: bool=False, verified_note_components: bool=False, verified_note_fact_credit: bool=False, verified_note_synthesis: bool=False, verified_note_synthesis_credit: bool=False, verified_note_assessment_credit: bool=False, partial_notebook: bool=False, referenced_notebook: bool=False, paired_notebook: bool=False, pair_action_notebook: bool=False, pair_draft_synthesis: bool=False, whole_reference_notebook: bool=False, whole_reference_pages: bool=False, context_limit: int=16384) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `type`, `sum`, `ValueError`, `load_protocol`, `protocol.pop`, `protocol.update`, `quantization_contract`, `AUTHORIZATION.copy`, `digest` 외 15개
- 반환 표현: `protocol`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`restore_protocol` — L235**
```python
restore_protocol(record: dict) -> dict
```
- 기존 설명: Only explicitly versioned complete contracts may resume across JSON boundaries.
- 직접 호출: `type`, `record.get`, `ValueError`, `build_protocol`, `json.loads`, `json.dumps`, `digest`
- 반환 표현: `protocol`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`local_prompts` — L277**
```python
local_prompts(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `protocol.get`, `ValueError`, `restore_protocol`, `load_prompts`, `notebook_synthesis.prompt_contract`, `SOURCE_OBSERVATION_INSTRUCTION.copy`, `result.update`, `generation.prompt_contract`, `action.prompt_contract`, `action.contract` 외 7개
- 반환 표현: `result`; `prompts | {'rewarded_notebook': NOTEBOOK_PROMPT, 'notebook_input_policy': INPUT_POLICY, 'notebook_route_rule': STRICT_ROUTE_RULE, 'notebook_feedback_policy': FEEDBACK_POLICY, 'notebook_ …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.103 `fine_tuning/notebook_rl/local_run.py`

[현재 소스](../../fine_tuning/notebook_rl/local_run.py) · SHA256 `9b60b86e10ddeeecc1f5f048e005677648ec1f2b84ce3e597aad0aac299cae53`

모듈의 기존 설명: Local real-model minimal RL and fresh-process reload, using the existing trainer.

**`write_json` — L27**
```python
write_json(path: Path, value) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_write`, `_bytes`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`cache_readiness` — L31**
```python
cache_readiness(path: Path, model_path: Path, protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(path / 'result.json').read_text`, `result.get`, `checkpoint.file_hash`, `checkpoint.digest`, `(path / 'protocol.json').read_text`, `(path / 'sources.json').read_text`, `_sources`, `ValueError`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`read_actor_descriptor` — L44**
```python
read_actor_descriptor(value: dict) -> list[ActorItem]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_read`, `descriptor`, `_sha`, `ValueError`, `json.loads`, `type`, `from_json`, `json.dumps`
- 반환 표현: `[from_json(ActorItem, json.dumps(row, allow_nan=False)) for row in rows]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`read_execution_actors` — L54**
```python
read_execution_actors(prepared: dict, originals: list[ActorItem], protocol: dict) -> list[ActorItem]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `project_observation_actors`, `prepared.get`, `ValueError`, `read_actor_descriptor`
- 반환 표현: `actors`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`read_preparation` — L66**
```python
read_preparation(path: Path, protocol: dict) -> tuple[dict, dict, list[ActorItem]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `path.read_text`, `prepared.get`, `ValueError`, `read_descriptor`, `request.get`, `str`, `Path`, `checked.get`, `approval.code_hashes` 외 10개
- 반환 표현: `(prepared, request, read_execution_actors(prepared, actors, protocol))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`minimal_actors` — L103**
```python
minimal_actors(actors: list[ActorItem]) -> list[ActorItem]
```
- 기존 설명: Fix one single-image and one multi-image problem before observing answers.
- 직접 호출: `len`, `ValueError`
- 반환 표현: `[singles[0], multiple[0]]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`parameter_hashes` — L112**
```python
parameter_hashes(bundle) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dict`, `bundle.model.named_parameters`, `tensor_sha256`
- 반환 표현: `{group: {r['name']: tensor_sha256(actual[r['name']]) for r in bundle.inventory['trainable'] if r['group'] == group} for group in ('L', 'M', 'V')}`

**`verify_changes` — L120**
```python
verify_changes(before: dict, after: dict, update: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `sum`, `any`, `update['routes'].values`
- 반환 표현: `{'changed_tensor_counts': changed, 'vision_on_observed': vision_on, 'vision_off_only_observed': not vision_on, 'both_routes_verified': False}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_binding` — L134**
```python
_binding(prepared: dict, protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `checkpoint.digest`, `approval.code_hashes`
- 반환 표현: `{'protocol_sha256': protocol['sha256'], 'prompts_sha256': protocol['prompts_sha256'], 'data_sha256': checkpoint.digest(prepared), 'code_sha256': checkpoint.digest(approval.code_hashes() …`

**`minimum_execution_contract` — L139**
```python
minimum_execution_contract(policy: str | None, protocol: dict) -> dict | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `restore_protocol`, `ValueError`
- 반환 표현: `None`; `{'version': 'minimum-execution-contract-v1', 'policy': policy, 'stage': 'MINIMAL_TWO_PROBLEM_ONE_UPDATE', 'max_optimizer_updates': 1, 'require_retained_learning_signal': True}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`minimum_binding` — L152**
```python
minimum_binding(prepared: dict, protocol: dict, execution_contract: dict | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_binding`, `type`, `minimum_execution_contract`, `execution_contract.get`, `ValueError`, `checkpoint.digest`
- 반환 표현: `_binding(prepared, protocol)`; `_binding(prepared, protocol) | {'data_sha256': checkpoint.digest({'preparation_sha256': checkpoint.digest(prepared), 'execution_contract_sha256': checkpoint.digest(execution_contract)}) …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`retained_update_signal` — L163**
```python
retained_update_signal(update: dict, protocol: dict) -> dict
```
- 기존 설명: Check recorded retained coefficients; actual update/reload evidence is separate.
- 직접 호출: `update.get`, `effective_mis_deltas`, `type`, `set`, `signal.get`, `any`, `math.isfinite`, `deltas.values`, `len`, `ValueError` 외 4개
- 반환 표현: `signal`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_install_actor_guards` — L192**
```python
_install_actor_guards(prepared: dict, request: dict, *, extra_vault_files=()) -> list[Path]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_versions`, `_prepare_hf_dependencies`, `import_module`, `Path`, `Path.home`, `install_guards`
- 반환 표현: `forbidden`

**`make_probes` — L211**
```python
make_probes(actors, images, engine, prompts) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `SyncProbe`, `RolloutRequest`, `user_message`, `turn1_text`, `len`, `engine.tokenizer.encode`, `minimal_actors`
- 반환 표현: `[SyncProbe(RolloutRequest('local-sync:' + a.problem_id, None, [user_message(turn1_text(a, prompts), len(images[a.problem_id]))], images[a.problem_id], 1, 8, [3407]), engine.tokenizer.en …`

**`probe_evidence` — L222**
```python
probe_evidence(engine, probes) -> dict
```
- 기존 설명: Actual single/multi-image forward and short generation, not task accuracy.
- 직접 호출: `engine.prepare`, `inputs.append`, `len`, `item.token_ids.count`, `item.pixels['image_grid_thw'].tolist`, `list`, `tensor_identity`, `item.pixels.items`, `logprobs.append`, `engine.probe_logprobs` 외 2개
- 반환 표현: `{'inputs': inputs, 'reference_logprobs': logprobs, 'generated': [asdict(result) for result in outputs], 'task_accuracy_measured': False}`

**`_positive_error_relationship` — L241**
```python
_positive_error_relationship(entry: dict, *, native_only: bool=False) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `entry.get`
- 반환 표현: `bool(entry['reward'] == 1 and entry['advantage'] > 0 and entry['token_indices'] and (entry.get('credit_type') == 'NATIVE_ERROR_RELATIONSHIP' and entry['relation'] in ('CORRECTED', 'PERS …`

**`relationship_learning_summary` — L247**
```python
relationship_learning_summary(update: dict) -> dict
```
- 기존 설명: Keep aggregate relationship effects distinct from measured native-only VJPs.
- 직접 호출: `set`, `row.get`, `gradient.get`, `learned.add`, `positive.update`, `native.get`, `type`, `all`, `math.isfinite`, `norms.values` 외 3개
- 반환 표현: `{'learned_signal_types': sorted(learned), 'positive_verified_relationship_credit_types': sorted(positive), 'output_finalization_only_gradient_verified': 'OUTPUT_FINALIZATION' in learned …`

**`native_relationship_update_evidence` — L280**
```python
native_relationship_update_evidence(update: dict, protocol: dict) -> dict
```
- 기존 설명: Bind native-only commit evidence to retained raw-token credits and actual VJPs.
- 직접 호출: `update.get`, `relationship_learning_summary`, `signal.get`, `effective_mis_deltas`, `protocol.get`, `set`, `type`, `any`, `len`, `ValueError` 외 9개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`error_relationship_signal` — L334**
```python
error_relationship_signal(trees: list, objective: dict | None, tokenizer, *, native_only: bool=False) -> bool
```
- 기존 설명: Require original-group variance on mapped, verified error-relationship units.
- 직접 호출: `synthesis_credit.contract`, `ValueError`, `notebook_objective_scores`, `synthesis_credit.group_credit`, `any`, `_positive_error_relationship`, `credits.values`
- 반환 표현: `True`; `False`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_positive_assessment` — L353**
```python
_positive_assessment(entry: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `type`, `entry.get`, `all`, `math.isfinite`
- 반환 표현: `bool(type(entry) is dict and type(entry.get('reward')) is int and (entry['reward'] == 1) and all((type(entry.get(key)) in (int, float) and math.isfinite(entry[key]) and (entry[key] > 0) …`

**`assessment_signal` — L360**
```python
assessment_signal(trees: list, objective: dict | None, tokenizer) -> bool
```
- 기존 설명: Use the original n4 assessment literals; relationships cannot satisfy this gate.
- 직접 호출: `synthesis_credit.contract`, `ValueError`, `notebook_objective_scores`, `synthesis_credit.group_credit`, `any`, `_positive_assessment`, `credits.values`
- 반환 표현: `True`; `False`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`assessment_update_evidence` — L377**
```python
assessment_update_evidence(update: dict, protocol: dict, *, trees=None, fresh_credits=None, parameter_inventory=None) -> dict
```
- 기존 설명: Validate exact signed aggregate membership; fresh replay supplies original trees/credits.
- 직접 호출: `update.get`, `protocol.get`, `synthesis_credit.contract`, `type`, `any`, `row.get`, `len`, `ValueError`, `aggregate.get`, `_hash` 외 48개
- 반환 표현: `{'assessment_learning_verified': True, 'assessment_only_gradient_verified': True, 'assessment_gradient_sample_ids': measured, 'retained_assessment_signal': signal, 'strict_notebook_corr …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`capture` — L572**
```python
capture(preparation: Path, model_path: Path, output: Path, *, standalone: bool, cache_check: Path | None=None, verified_note_atoms: bool=False, verified_note_components: bool=False, verified_note_fact_credit: bool=False, verified_note_synthesis: bool=False, verified_note_synthesis_credit: bool=False, verified_note_assessment_credit: bool=False, partial_notebook: bool=False, referenced_notebook: bool=False, paired_notebook: bool=False, pair_action_notebook: bool=False, pair_draft_synthesis: bool=False, whole_reference_notebook: bool=False, whole_reference_pages: bool=False, context_limit: int=16384, execution_policy: str | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `build_protocol`, `protocol.get`, `minimum_execution_contract`, `ValueError`, `cache_readiness`, `read_preparation`, `minimal_actors`, `Path`, `local_prompts`, `minimum_binding` 외 65개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`reload_capture` — L787**
```python
reload_capture(capture_root: Path, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(capture_root / 'launch.json').read_text`, `(capture_root / 'result.json').read_text`, `(capture_root / 'reload_expected.json').read_text`, `type`, `launch.get`, `os.getpid`, `psutil.pid_exists`, `ValueError`, `restore_protocol` 외 40개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L879**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_subparsers`, `sub.add_parser`, `prepare.add_argument`, `prepare.add_mutually_exclusive_group`, `prepare_generation.add_argument`, `prepare_reference.add_argument`, `prepare_objective.add_argument`, `run.add_argument`, `run.add_mutually_exclusive_group` 외 15개
- 반환 표현: `0 if result['status'] == 'LOCAL_DATA_REVALIDATED' else 2`; `1`; `0`

### D18.104 `fine_tuning/notebook_rl/local_train.py`

[현재 소스](../../fine_tuning/notebook_rl/local_train.py) · SHA256 `c98c045df00144c9b331b628b3fe3a99c71cf607c2bcc138eb90c6ee4425de96`

모듈의 기존 설명: Fixed-ten local RL and matched notebook-reward control after the minimal gate.

**`execution_max_seconds` — L34**
```python
execution_max_seconds(contract: dict, protocol: dict) -> int
```
- 기존 설명: Only the explicitly bound fixed-ten training budget may override the protocol default.
- 직접 호출: `contract.get`, `type`, `set`, `ValueError`
- 반환 표현: `budget['max_seconds']`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`execution_admission` — L45**
```python
execution_admission(contract: dict, protocol: dict) -> str
```
- 기존 설명: Historical contracts remain strict; exploration requires the exact V16 objective.
- 직접 호출: `contract.get`, `protocol.get`, `synthesis_contract`, `ValueError`, `restore_protocol`, `protocol.get('notebook_objective', {}).get`, `minimum_execution_contract`, `restore_protocol(protocol).get`, `objective_contract`
- 반환 표현: `admission`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_read` — L97**
```python
_read(path: Path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `guarded_read`
- 반환 표현: `json.loads(guarded_read(path), object_pairs_hook=_unique)`

**`notebook_learning_gate` — L103**
```python
notebook_learning_gate(capture: Path, result: dict, update: dict, protocol: dict) -> list[Path]
```
- 기존 설명: Bind a rewarded note to its isolated backward pass, not merely answer learning.
- 직접 호출: `protocol.get('notebook_objective', {}).get`, `protocol.get`, `ValueError`, `native_relationship_update_evidence`, `type`, `result.get`, `update.get`, `len`, `objective.get`, `set` 외 35개
- 반환 표현: `paths`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`parent_gate` — L244**
```python
parent_gate(capture: Path, reload: Path, *, require_notebook_learning: bool=False) -> dict
```
- 기존 설명: Recheck closed real-update evidence; keep its historical code binding intact.
- 직접 호출: `Path(capture).absolute`, `Path`, `Path(reload).absolute`, `_read`, `json.loads`, `json.dumps`, `restore_protocol`, `launch.get`, `type`, `psutil.pid_exists` 외 19개
- 반환 표현: `{'status': 'REAL_MINIMAL_UPDATE_AND_FRESH_RELOAD_VERIFIED', 'launch': launch, 'notebook_reward_learning_verified': require_notebook_learning, 'files': [{'path': str(p), 'sha256': checkp …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fact_credit_learning_gate` — L299**
```python
fact_credit_learning_gate(capture: Path, result: dict, update: dict, protocol: dict, *, actors: dict, verifier, tokenizer, replay_relation_v1: bool=False, admission: str | None=None, tree_ids: set[str] | None=None, control: bool=False, parameter_inventory=None) -> list[Path]
```
- 기존 설명: Regrade complete groups before MIS; require retained positive fact-token learning.
- 직접 호출: `protocol.get('notebook_objective', {}).get`, `protocol.get`, `ValueError`, `update.get`, `retained_update_signal`, `native_relationship_update_evidence`, `execution_admission`, `result.get`, `effective_mis_deltas`, `len` 외 47개
- 반환 표현: `paths`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`training_parent_gate` — L506**
```python
training_parent_gate(capture: Path, reload: Path, *, admission: str=STRICT_ADMISSION) -> dict
```
- 기존 설명: Exploratory functional admission never certifies strict notebook correctness.
- 직접 호출: `parent_gate`, `ValueError`, `Path(capture).absolute`, `Path`, `restore_protocol`, `execution_admission`, `_read`, `launch.get`, `type`, `checkpoint.digest` 외 37개
- 반환 표현: `parent_gate(capture, reload, require_notebook_learning=True)`; `gate | {'admission_policy': admission, 'verified_update_exploration_admitted': True, 'retained_policy_coefficient_signal_verified': True, 'strict_notebook_correctness_verified': False,  …`; `gate | compatibility | summary | {'admission_policy': admission, verified_key: True, 'strict_notebook_correctness_verified': False, 'files': gate['files'] + [{'path': str(p), 'sha256':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fixed_pairs` — L647**
```python
fixed_pairs(actors: list) -> list[list]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `ValueError`, `range`
- 반환 표현: `[actors[i:i + 2] for i in range(0, 10, 2)]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`eligible_trees` — L653**
```python
eligible_trees(trees: list, arm: str, *, notebook_objective: dict | None=None) -> list
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `partition_learning_trees`
- 반환 표현: `partition_learning_trees(trees, include_notebooks=arm == 'notebook', notebook_objective=notebook_objective)[0]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`training_binding` — L662**
```python
training_binding(prepared: dict, protocol: dict, contract: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `execution_max_seconds`, `_binding`, `checkpoint.digest`
- 반환 표현: `_binding(prepared, protocol) | {'data_sha256': checkpoint.digest({'preparation_sha256': checkpoint.digest(prepared), 'execution_contract_sha256': checkpoint.digest(contract)})}`

**`native_collection_gate` — L669**
```python
native_collection_gate(prepared: dict, protocol: dict, readiness: dict, actors: list) -> dict
```
- 기존 설명: Permission to collect the original cohort is separate from any learning evidence.
- 직접 호출: `execution_admission`, `read_descriptor`, `fixed_pairs`, `original.get`, `selected.get`, `ValueError`, `checkpoint.digest`, `approval.code_hashes`
- 반환 표현: `{'status': 'FIXED_TEN_NATIVE_COLLECTION_ADMITTED', 'collection_allowed': True, 'admission_policy': FIXED_TEN_NATIVE_EXPLORATION, 'preparation_sha256': checkpoint.digest(prepared), 'cach …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_native_collection` — L691**
```python
validate_native_collection(contract: dict, prepared: dict, protocol: dict, readiness: dict, actors: list) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `native_collection_gate`, `contract.get`, `type`, `ValueError`, `execution_max_seconds`
- 반환 표현: `gate`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`assessment_control_update_evidence` — L710**
```python
assessment_control_update_evidence(update: dict, protocol: dict, trees: list) -> None
```
- 기존 설명: The matched control keeps DIAG actions for KL and removes their reward credit.
- 직접 호출: `protocol.get`, `update.get`, `any`, `update.get('effective_rewards', {}).get`, `len`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_native_updates` — L729**
```python
validate_native_updates(capture: Path, result: dict, contract: dict, protocol: dict, *, parameter_inventory=None) -> None
```
- 기존 설명: Every counted commit retains its own original pair's token/VJP evidence.
- 직접 호출: `protocol.get`, `execution_admission`, `contract.get`, `ValueError`, `sorted`, `capture.glob`, `from_json`, `json.dumps`, `(record := _read(path)).get`, `_read` 외 36개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`matched_notebook_arm` — L860**
```python
matched_notebook_arm(path: Path, prepared: dict, protocol: dict, model_path: Path, cache_check: Path, pairs: list[list], *, historical_code_sha256: str | None=None, admission: str=STRICT_ADMISSION, parent: dict | None=None, max_seconds: int | None=None) -> dict
```
- 기존 설명: Training requires current code; read-only evaluation may pin its verified training source.
- 직접 호출: `ValueError`, `type`, `re.fullmatch`, `_read`, `launch.get`, `execution_max_seconds`, `execution_admission`, `contract.get`, `training_binding`, `result.get` 외 7개
- 반환 표현: `{'initialization': {'path': str(path), 'sha256': checkpoint.file_hash(path)}, 'launch_sha256': checkpoint.file_hash(path.parent / 'launch.json'), 'result_sha256': checkpoint.file_hash(p …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`train` — L904**
```python
train(preparation: Path, model_path: Path, cache_check: Path, minimal: Path | None, minimal_reload: Path | None, output: Path, *, arm: str, match_initialization: Path | None=None, admission: str=STRICT_ADMISSION, max_seconds: int=14400) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `build_protocol`, `training_parent_gate`, `restore_protocol`, `execution_admission`, `execution_max_seconds`, `protocol.get`, `str`, `cache_readiness`, `read_preparation` 외 78개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`reload_training` — L1192**
```python
reload_training(capture: Path, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_read`, `restore_protocol`, `execution_admission`, `checkpoint.digest`, `ValueError`, `launch.get`, `os.getpid`, `type`, `psutil.pid_exists`, `json.loads` 외 39개
- 반환 표현: `restored`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L1294**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_subparsers`, `sub.add_parser`, `run.add_argument`, `reload.add_argument`, `parser.parse_args`, `train`, `reload_training`, `print`, `json.dumps`
- 반환 표현: `0 if result['status'] in ('LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD', 'LOCAL_TEN_FRESH_PROCESS_RELOAD_VERIFIED') else 2`

### D18.105 `fine_tuning/notebook_rl/losses.py`

[현재 소스](../../fine_tuning/notebook_rl/losses.py) · SHA256 `510302654c71622e766cf61821f3abdef69dec6c87a068464d8b7b64bc604bd6`

모듈의 기존 설명: Pure Torch v2 group-relative token losses; no model or gold-label forward.

**`_finite_tensor` — L10**
```python
_finite_tensor(value: torch.Tensor, name: str, *, floating: bool=True) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `value.is_floating_point`, `value.is_complex`, `bool`, `torch.isfinite(value).all`, `torch.isfinite`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_scalar` — L19**
```python
_scalar(value: float, name: str, *, minimum: float=0.0, positive: bool=False) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `math.isfinite`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_matching` — L25**
```python
_matching(left: torch.Tensor, right: torch.Tensor, name: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_finite_tensor`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_actions` — L31**
```python
_actions(action_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_finite_tensor`, `ValueError`, `bool`, `((action_mask == 0) | (action_mask == 1)).all`, `action_mask.detach().bool`, `action_mask.detach`, `mask.any(dim=1).all`, `mask.any`
- 반환 표현: `mask`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_matrix` — L43**
```python
_matrix(value: torch.Tensor, name: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_finite_tensor`, `any`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_work` — L49**
```python
_work(value: torch.Tensor) -> torch.Tensor
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `value.float`
- 반환 표현: `value.float() if value.dtype in (torch.float16, torch.bfloat16) else value`

**`group_advantages` — L54**
```python
group_advantages(rewards: torch.Tensor) -> torch.Tensor
```
- 기존 설명: Center a complete group's rewards, with no standard-deviation scaling.
- 직접 호출: `_finite_tensor`, `rewards.numel`, `ValueError`, `_work`, `rewards.detach`, `rewards.mean`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`mis_mask` — L65**
```python
mis_mask(old_lp: torch.Tensor, samp_lp: torch.Tensor, action_mask: torch.Tensor, delta_seq: float=0.05) -> tuple[torch.Tensor, torch.Tensor]
```
- 기존 설명: Keep sequences whose action-token mean log ratio is within delta_seq.
- 직접 호출: `_matrix`, `_matching`, `_scalar`, `_actions`, `_work`, `old_lp.detach`, `samp_lp.detach`, `torch.where(mask, delta, 0.0).sum`, `torch.where`, `mask.sum` 외 2개
- 반환 표현: `(mean_delta.abs() <= delta_seq, mean_delta)`

**`k3` — L78**
```python
k3(ref_lp: torch.Tensor, cur_lp: torch.Tensor) -> torch.Tensor
```
- 기존 설명: Elementwise k3 KL estimate; only the current policy receives gradients.
- 직접 호출: `_finite_tensor`, `_matching`, `_work`, `ref_lp.detach`, `(torch.expm1(delta) - delta).clamp_min`, `torch.expm1`
- 반환 표현: `result`

**`group_loss` — L88**
```python
group_loss(cur_lp: torch.Tensor, old_lp: torch.Tensor, ref_lp: torch.Tensor, adv: torch.Tensor, action_mask: torch.Tensor, seq_keep: torch.Tensor, *, eps_low: float, eps_high: float, beta: float, norm_const: float, partial_adv: torch.Tensor | None=None) -> torch.Tensor
```
- 기존 설명: Return a group (or sample-slice) sum divided by the full n_g * L_cap.
- 직접 호출: `_matrix`, `_matching`, `_finite_tensor`, `ValueError`, `_scalar`, `isinstance`, `_actions`, `seq_keep.detach().unsqueeze`, `seq_keep.detach`, `_work` 외 11개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`combine` — L136**
```python
combine(group_losses: list[torch.Tensor], lambdas: list[float]) -> torch.Tensor
```
- 기존 설명: Combine complete usable groups once, using the step's fixed weight sum.
- 직접 호출: `len`, `ValueError`, `_finite_tensor`, `_scalar`, `sum`, `torch.stack([loss * weight for loss, weight in zip(group_losses, lambdas, strict=True)]).s …`, `torch.stack`, `zip`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.106 `fine_tuning/notebook_rl/measure.py`

[현재 소스](../../fine_tuning/notebook_rl/measure.py) · SHA256 `011cbf9e29b3a723e71b607a6ccefee537cce8c72b4e60fa8edec1f3b9e59b45`

모듈의 기존 설명: Run fixed-F0 candidate measurements; never approve data or start training.

**`validate_job` — L26**
```python
validate_job(job: dict, protocol: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `isinstance`, `re.fullmatch`, `ValueError`, `utc`, `launch.descriptor`, `Path(job[key]).is_absolute`, `Path`, `any` 외 7개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`preflight` — L64**
```python
preflight(job: dict, protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_job`, `approval.validate_for_measurement`, `Path`
- 반환 표현: `approval.validate_for_measurement(Path(job['build_root']), **{k: job[k] for k in (*DESCRIPTORS, 'expected_measurement_scope')})`

**`execute` — L70**
```python
execute(job: dict, protocol: dict, validated: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validated.get`, `ValueError`, `preflight`, `launch.checkpoint_binding`, `launch.runtime_check`, `Path`, `_safe_root`, `output.mkdir`, `_write`, `_bytes` 외 12개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`main` — L130**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_mutually_exclusive_group`, `mode.add_argument`, `parser.parse_args`, `launch.read_json`, `load_protocol`, `preflight`, `print`, `json.dumps` 외 3개
- 반환 표현: `0 if checked['status'] == 'VALIDATED_FOR_MEASUREMENT' else 2`; `0 if result['status'] == 'MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL' else 2`; `2`

### D18.107 `fine_tuning/notebook_rl/metrics.py`

[현재 소스](../../fine_tuning/notebook_rl/metrics.py) · SHA256 `560ed5015c1425fcb35cb716750e098a38d0267b194900ece99c3fafe41496c9`

모듈의 기존 설명: Finite, append-only operational evidence and predeclared health thresholds.

**`step_record` — L17**
```python
step_record(**fields) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `fields.keys`, `ValueError`, `type`, `finite`, `set`, `fields['grad_norm'].values`, `fields['seconds'].values`, `json.loads`, `json.dumps`
- 반환 표현: `json.loads(json.dumps(fields, allow_nan=False))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`health_check` — L55**
```python
health_check(history: list[dict], protocol: dict, f0_reference: dict) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `step_record`, `any`, `zip`, `ValueError`, `above`, `stops.append`, `f0_reference.get`, `type`, `math.isfinite`, `len` 외 1개
- 반환 표현: `[]`; `stops`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`append_record` — L91**
```python
append_record(path: Path, record: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `(json.dumps(step_record(**record), sort_keys=True, allow_nan=False) + '\n').encode`, `json.dumps`, `step_record`, `path.is_symlink`, `ValueError`, `path.exists`, `path.read_bytes`, `raw.endswith`, `json.loads`, `raw.splitlines` 외 5개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.108 `fine_tuning/notebook_rl/monitor.py`

[현재 소스](../../fine_tuning/notebook_rl/monitor.py) · SHA256 `4e268af4ee88a390e98145a243cca9ddea5f4353e05f28ebc4425a28d909c91e`

모듈의 기존 설명: Matched dev QA and independent observation health measurements, never training.

**클래스 `_Incomplete` — L39**

기저: `Exception`.

**`_json` — L43**
```python
_json(value) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ': …`, `json.dumps`
- 반환 표현: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()`

**`_sha` — L47**
```python
_sha(raw: bytes) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`
- 반환 표현: `hashlib.sha256(raw).hexdigest()`

**`_hash` — L51**
```python
_hash(value) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_json`
- 반환 표현: `_sha(_json(value))`

**`_source_hashes` — L55**
```python
_source_hashes() -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_read`
- 반환 표현: `{name: _sha(_read(_HERE / name)) for name in _SOURCE}`

**`_configuration` — L59**
```python
_configuration(actor_sha, vault_sha, protocol, prompts, input_binding, seed, n) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_hash`, `dict`, `asdict`, `_source_hashes`
- 반환 표현: `{'format': VERSION, 'n': n, 'samples_per_problem': 1, 'protocol_sha256': protocol['sha256'], 'resolved_protocol_sha256': _hash(protocol), 'prompts_sha256': prompts['sha256'], 'actor_inp …`

**`_image_binding` — L68**
```python
_image_binding(actor, images) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `_sha`, `image.tobytes`, `zip`
- 반환 표현: `[{'sha256': sha, 'size': list(image.size), 'mode': image.mode, 'pixels_sha256': _sha(image.tobytes())} for sha, image in zip(actor.image_sha256, images, strict=True)]`

**`_frozen` — L73**
```python
_frozen(protocol: dict, input_binding: dict, seed: int) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `any`, `isinstance`, `len`, `input_binding.values`, `ValueError`, `load_protocol`, `load_prompts`
- 반환 표현: `prompts`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_require` — L87**
```python
_require(condition, code: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_Incomplete`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `_Incomplete`; 전파되는 callee 예외는 별도.

**`_write` — L92**
```python
_write(path: Path, value: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.open`, `os.fchmod`, `stream.fileno`, `stream.write`, `_json`, `stream.flush`, `os.fsync`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_append` — L100**
```python
_append(stream, value: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `stream.write`, `_json`, `stream.flush`, `os.fsync`, `stream.fileno`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_aggregate` — L106**
```python
_aggregate(rows: list[dict], n: int) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `len`, `from_json`, `_json(r['result']).decode`, `_json`, `all`, `sum`, `statistics.median`
- 반환 표현: `{'dev_first_attempt_acc': sum((r['result']['status'] == 'CORRECT' for r in qa)) / n, 'grounding_probe': correct / required, 'grounding_counts': {'required_correct': correct, 'required_t …`

**`_reference` — L124**
```python
_reference(path: Path, n: int) -> tuple[dict, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_require`, `_safe_root`, `_read`, `json.loads`, `reference.get`, `type`, `_hash`, `reference.items`, `set`, `reference['files'].items` 외 8개
- 반환 표현: `(reference, _sha(raw))`

**`_reference_ledgers` — L145**
```python
_reference_ledgers(reference: dict, rows: list[dict], root: Path, n: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `_read(root / 'sampler_calls.jsonl').splitlines`, `_read`, `_require`, `len`, `reference.get`, `all`, `manifest.get`, `configuration.items`, `verifier.get` 외 7개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`validate_f0_reference` — L190**
```python
validate_f0_reference(path: Path, split_root: Path, image_root: Path, *, protocol: dict, input_binding: dict, seed: int=3407) -> dict
```
- 기존 설명: Read-only dev96 preallocation check; raises ValueError, never creates approval.
- 직접 호출: `_validate_reference`, `ValueError`, `isinstance`, `str`, `type`
- 반환 표현: `_validate_reference(path, split_root, image_root, protocol=protocol, input_binding=input_binding, seed=seed, _expected_n=96)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_reference` — L205**
```python
_validate_reference(path, split_root, image_root, *, protocol, input_binding, seed, _expected_n) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_frozen`, `_safe_root`, `Path`, `sorted`, `read_actors`, `_require`, `len`, `_sha`, `_read`, `_configuration` 외 12개
- 반환 표현: `{'status': 'VALIDATED_F0_REFERENCE', 'n': _expected_n, 'samples_per_problem': 1, 'protocol_sha256': protocol['sha256'], 'actor_inputs_sha256': actor_sha, 'measurement_sha256': reference …`

**`measure_dev` — L238**
```python
measure_dev(split_root: Path, image_root: Path, output: Path, *, engine: Any, protocol: dict, step: int, deadline_timestamp: float, input_binding: dict, f0_report: Path | None=None, seed: int=3407, forbidden_roots=(), clock: Callable[[], float]=time.time) -> dict
```
- 기존 설명: Production entry point: fixed dev96, no exposed small-count/test verifier override.
- 직접 호출: `_measure`
- 반환 표현: `_measure(split_root, image_root, output, engine=engine, protocol=protocol, step=step, deadline_timestamp=deadline_timestamp, input_binding=input_binding, f0_report=f0_report, seed=seed, …`

**`_measure` — L247**
```python
_measure(split_root: Path, image_root: Path, output: Path, *, engine: Any, protocol: dict, step: int, deadline_timestamp: float, input_binding: dict, f0_report: Path | None, seed: int, forbidden_roots, clock: Callable[[], float], _expected_n: int) -> dict
```
- 기존 설명: Private small-fixture harness; public measure_dev always supplies 96.
- 직접 호출: `Path`, `type`, `math.isfinite`, `ValueError`, `_frozen`, `sorted`, `read_actors`, `len`, `getattr`, `callable` 외 55개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

### D18.109 `fine_tuning/notebook_rl/notebook_draft_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_draft_probe.py) · SHA256 `0f6c02f4fe26ef3a1980a37e222dd6195d766f7617951155906014fac0abbf9a`

모듈의 기존 설명: Independent per-attempt drafts followed by one unchanged-checker notebook diagnosis.

**`contract` — L60**
```python
contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'protocol_id': VERSION, 'runtime_protocol_sha256': protocol['sha256'], 'draft_prompt': DRAFT_PROMPT, 'cause_rules': CAUSE_RULES, 'final_suffix': FINAL_SUFFIX, 'final_base_prompt': note …`

**`draft_messages` — L74**
```python
draft_messages(item, index: int) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `notebook.public_transcript`, `notebook.solution_samples`, `public.items`, `attempt.update`, `len`, `DRAFT_PROMPT.format`, `layout._escaped`, `json.dumps`, `user_message`, `p0_body` 외 1개
- 반환 표현: `[user_message(p0_body(item.actor.question, item.actor.choices) + '\n\n' + fresh_observation.quoted_context(item.tree.notebook['rewarded_notebook']['fresh_observation']) + body, len(item …`

**`final_messages` — L86**
```python
final_messages(item, drafts: list[dict]) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `FINAL_SUFFIX.format`, `layout._escaped`, `json.dumps`, `user_message`, `len`
- 반환 표현: `[user_message(text, len(item.actor.image_paths))]`

**`_binding` — L94**
```python
_binding(item, engine, verifier) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `notebook.audit_episode`, `ValueError`, `layout._messages`, `next`, `seeds`, `notebook.solution_samples`, `note.get`, `_hash`, `len`, `fresh_observation.validate` 외 5개
- 반환 표현: `(audit, layout._image_identity(actor, original))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`collect` — L129**
```python
collect(requests, engine, verifier, record, deadline: float) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `approval.code_hashes`, `time.monotonic`, `TimeoutError`, `ValueError`, `seen.add`, `to_json`, `_hash`, `vars`, `_binding` 외 10개
- 반환 표현: `rows`
- 명시적 예외: `TimeoutError`, `ValueError`; 전파되는 callee 예외는 별도.

### D18.110 `fine_tuning/notebook_rl/notebook_layout_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_layout_probe.py) · SHA256 `aa48a541aad1f80122bd53bd4f10e4789685d74ab5ceb2b87fd52a7a4354a840`

모듈의 기존 설명: Paired notebook-input layout diagnosis; no training, repairs, or reward changes.

**`contract` — L23**
```python
contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`
- 반환 표현: `{'protocol_id': VERSION, 'runtime_protocol_sha256': protocol['sha256'], 'conditions': list(CONDITIONS), 'n': 4, 'max_new_tokens': MAX_NEW_TOKENS, 'context_limit': CONTEXT_LIMIT, 'max_se …`

**`trained_contract` — L35**
```python
trained_contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `contract`
- 반환 표현: `{**contract(protocol), 'protocol_id': TRAINED_VERSION, 'conditions': ['CURRENT_JSON'], 'order': 'original-fixed-actor-order', 'input_intervention': 'restored post-update weights on fixe …`

**클래스 `ProbeRequest` — L44**

기저: `object`.

```python
actor: ActorItem
tree: ProblemTree
condition: str
request: RolloutRequest
schema: str
control_messages: list[dict]
```

**`_escaped` — L53**
```python
_escaped(text: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `text.replace('<', '\\u003c').replace`, `text.replace`
- 반환 표현: `text.replace('<', '\\u003c').replace('>', '\\u003e')`

**`numbered_transcript` — L57**
```python
numbered_transcript(samples) -> str
```
- 기존 설명: Each physical source line remains a reversible, separately quoted string.
- 직접 호출: `zip`, `notebook.public_transcript`, `row.items`, `len`, `_numbered_first_attempt`, `lines.replace`, `ord`, `rows.append`, `json.dumps`, `_escaped` 외 1개
- 반환 표현: `_escaped('\n'.join(rows))`

**`numbered_instruction` — L72**
```python
numbered_instruction(samples, audit: dict, schema: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `notebook.PROMPT.replace`, `template.replace`, `_escaped`, `json.dumps`, `notebook.public_observation_differences`, `template.format`, `numbered_transcript`
- 반환 표현: `template.format(observation_instruction=schema, success_attempt=audit['success_attempt'], transcript=numbered_transcript(samples), observation_differences=differences)`

**`_messages` — L85**
```python
_messages(actor, tree, schema, audit, condition) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `notebook.solution_samples`, `notebook.instruction`, `numbered_instruction`, `user_message`, `p0_body`, `fresh_observation.quoted_context`, `len`
- 반환 표현: `[user_message(p0_body(actor.question, actor.choices) + '\n\n' + fresh_observation.quoted_context(reread) + body, len(actor.image_paths))]`

**`paired_requests` — L94**
```python
paired_requests(actors, trees, images, prompts) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `ValueError`, `enumerate`, `zip`, `notebook.solution_samples`, `bool`, `notebook.solution_skips`, `all`, `skipped.append`, `for_actor` 외 10개
- 반환 표현: `{'requests': requests, 'skipped': skipped}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_image_identity` — L137**
```python
_image_identity(actor, prepared) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `tensor_identity`, `prepared.pixels.items`
- 반환 표현: `{'image_paths': list(actor.image_paths), 'image_sha256': list(actor.image_sha256), 'image_key': prepared.image_key, 'processed_sizes': [list(im.size) for im in prepared.images], 'pixel_ …`

**`collect` — L146**
```python
collect(requests, engine, verifier, record, deadline: float, *, generation_policy_version: int | None=None) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `time.monotonic`, `TimeoutError`, `check_policy`, `notebook.audit_episode`, `_messages`, `next`, `note.get`, `_hash` 외 19개
- 반환 표현: `rows`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

### D18.111 `fine_tuning/notebook_rl/notebook_learning.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_learning.py) · SHA256 `57de8361303af9d8ce5adffe42f6065288054da0198b67ff5fd6e3c911d296c1`

모듈의 기존 설명: Explicit local notebook RL: program-checkable claims, never notebook SFT targets.

**`_unique` — L60**
```python
_unique(pairs) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_first_attempt_lines` — L69**
```python
_first_attempt_lines(text: str) -> list[dict]
```
- 기존 설명: Lossless Unicode-character offsets; never normalize or strip line endings.
- 직접 호출: `enumerate`, `text.splitlines`, `spans.append`, `len`
- 반환 표현: `spans`

**`_numbered_first_attempt` — L78**
```python
_numbered_first_attempt(text: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `'\n'.join`, `json.dumps`, `_first_attempt_lines`
- 반환 표현: `'\n'.join((f"{row['line_id']}: {json.dumps(row['text'], ensure_ascii=False)}" for row in _first_attempt_lines(text)))`

**`_parse` — L83**
```python
_parse(text: str) -> dict | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `_validate`
- 반환 표현: `None`; `_validate(note)`

**`_validate` — L92**
```python
_validate(note: Any) -> dict | None
```
- 기존 설명: Validate parsed types before canonicalizing only finite answer numbers.
- 직접 호출: `type`, `set`, `str`, `number`, `any`, `note[k].strip`, `note['previous_answer'].strip`, `zip`, `k.strip`, `v.strip` 외 5개
- 반환 표현: `None`; `note`

**`_grade` — L128**
```python
_grade(problem_id: str, text: str, finish_reason: str, first: Sample, parent: Sample, target: DiagTarget | None, verifier: Any) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `hashlib.sha256(first.text.encode('utf-8')).hexdigest`, `hashlib.sha256`, `first.text.encode`, `_parse`, `_first_attempt_lines`, `len`, `bool`, `any`, `row['text'].strip` 외 7개
- 반환 표현: `record`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_recovered_path` — L179**
```python
_recovered_path(tree: ProblemTree, verifier: Any) -> tuple[Sample, Sample, DiagTarget | None] | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `samples.values`, `_hash`, `samples.get`, `len`, `ValueError`, `verifier.grade`, `verifier.diagnosis_target`
- 반환 표현: `(first, parent, target)`; `unknown`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`append_rewarded_notebooks` — L209**
```python
append_rewarded_notebooks(trees: list[ProblemTree], images: dict[str, list], engine: Any, verifier: Any, prompts: dict, cfg: RolloutConfig, *, actors: dict[str, ActorItem], on_group: Callable[[Group], None] | None=None, on_observation: Callable[[dict], None] | None=None, on_draft: Callable[[dict], None] | None=None, on_page: Callable[[dict], None] | None=None, should_stop: Callable[[], bool] | None=None) -> dict
```
- 기존 설명: Append on-policy DIAG actions for an explicitly selected local reward protocol.
- 직접 호출: `prompts.get`, `append_episode_notebooks`, `ValueError`, `cfg.validate`, `len`, `set`, `any`, `actors.items`, `to_json`, `validate_actor` 외 30개
- 반환 표현: `append_episode_notebooks(trees, images, engine, verifier, prompts, cfg, actors=actors, on_group=on_group, on_observation=on_observation, should_stop=should_stop, **{'on_draft': on_draft …`; `report`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.112 `fine_tuning/notebook_rl/notebook_objective.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_objective.py) · SHA256 `4543cfe093a74e2d4e4b90ff26feef97662876003d28368a5790560e6b8558d8`

모듈의 기존 설명: Host-only verified note atoms; the original strict reward remains unchanged.

**`contract` — L19**
```python
contract(version: str=VERSION) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `synthesis_contract`, `ValueError`, `base.update`
- 반환 표현: `synthesis_contract(version)`; `base`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`effective_reward` — L66**
```python
effective_reward(strict_reward: float, numerator: int, denominator: int) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `float`
- 반환 표현: `float(strict_reward + ATOMIC_WEIGHT * (numerator / denominator if denominator else 0.0))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`score_note` — L74**
```python
score_note(text, finish_reason, tree, audit, verifier, strict_report=None, *, actor, objective_version: str=VERSION, replay_relation_v1: bool=False) -> dict
```
- 기존 설명: Recompute the bound strict report; never trust or mutate a supplied record.
- 직접 호출: `ValueError`, `contract`, `notebook.audit_episode`, `audit.get`, `digest`, `tree.notebook.get('rewarded_notebook', {}).get`, `tree.notebook.get`, `notebook.grade_note`, `verifier.grade`, `notebook.solution_samples` 외 22개
- 반환 표현: `receipt`; `receipt | {'atomic_gate': 'ANSWER_UNVERIFIED'}`; `receipt | {'atomic_gate': 'INVALID_SLOT_IDS'}`; 외 5개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.113 `fine_tuning/notebook_rl/notebook_pair_drafts.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_pair_drafts.py) · SHA256 `dc9a6986b41cfb49327f5a92f75f38ccd049bbc53af46deba91d6c0b68904b25`

모듈의 기존 설명: Unverified pair proposals for a later whole-history notebook, never rewarded actions.

**`contract` — L25**
```python
contract() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_hash`
- 반환 표현: `{'version': VERSION, 'prompt': probe.PROMPT, 'prompt_sha256': _hash(probe.PROMPT), 'n': 4, 'max_new_tokens': MAX_NEW_TOKENS, 'context_limit': CONTEXT_LIMIT, 'seed_stage': 'PAIR_DRAFT:{o …`

**`_source` — L41**
```python
_source(actor, tree, schema) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `episode.solution_samples`, `episode.solution_skips`, `any`, `type`, `schema.strip`, `ValueError`, `_hash`, `json.loads`, `to_json`, `episode.public_transcript` 외 1개
- 반환 표현: `_hash({'actor': json.loads(to_json(actor)), 'schema': schema, 'policy_version': tree.policy_version, 'transcript': episode.public_transcript(samples), 'original_inputs': [{key: getattr( …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_plan` — L56**
```python
_plan(samples, reread) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `probe.select_targets`, `str`
- 반환 표현: `(probe.select_targets(samples, reread), None)`; `([], str(exc))`
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

**`_request` — L66**
```python
_request(actor, tree, reread, schema, images, master_seed, index, target) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `p0_body(actor.question, actor.choices).removesuffix`, `p0_body`, `user_message`, `fresh_observation.quoted_context`, `probe.instruction`, `episode.solution_samples`, `len`, `RolloutRequest`, `seeds`
- 반환 표현: `RolloutRequest(f'{actor.problem_id}:v{tree.policy_version}:PAIR_DRAFT:{index}', None, messages, images, 4, MAX_NEW_TOKENS, seeds(actor.problem_id, tree.policy_version, 1, f'PAIR_DRAFT:{ …`

**`_inputs` — L76**
```python
_inputs(request, index, target, actor, prepared=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `copy.deepcopy`, `_hash`, `list`, `json.loads`, `json.dumps`, `_image_identity`
- 반환 표현: `{'request_id': request.request_id, 'target_index': index, 'target': copy.deepcopy(target), 'messages': copy.deepcopy(request.messages), 'messages_sha256': _hash(request.messages), 'seed …`

**`_sealed` — L86**
```python
_sealed(record) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `record.get`, `_hash`, `contract`, `any`, `_ROLES.items`, `record.items`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_completions` — L94**
```python
_completions(values, tokenizer=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `len`, `ValueError`, `set`, `_completion`, `tokenizer.decode`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate` — L107**
```python
validate(record, actor, tree, reread, schema, *, tokenizer=None, preparation=None, images=None) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sealed`, `record.get`, `json.loads`, `to_json`, `type`, `_source`, `_hash`, `ValueError`, `fresh_observation.validate`, `_plan` 외 15개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`generate` — L173**
```python
generate(actor, tree, reread, schema, images, engine, *, master_seed=3407, context_limit=24576, on_event=None, should_stop=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_source`, `type`, `ValueError`, `fresh_observation.validate`, `_plan`, `episode.solution_samples`, `contract`, `json.loads`, `to_json`, `_hash` 외 16개
- 반환 표현: `record`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`quoted_context` — L234**
```python
quoted_context(record) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sealed`, `len`, `json.dumps(public, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ' …`, `json.dumps(public, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ' …`, `json.dumps`
- 반환 표현: `'The following pair proposals are quoted UNVERIFIED model outputs, including every invalid, truncated or contradictory draft. They are not verifier conclusions, expected relationships,  …`

### D18.114 `fine_tuning/notebook_rl/notebook_pair_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_pair_probe.py) · SHA256 `75180ab6ad54113fce9d4900ff1c0a8979e33bbeafc7f773a08e084d80be691f`

모듈의 기존 설명: Zero-update single-pair semantic diagnosis on the unchanged original history.

**`contract` — L44**
```python
contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `protocol.get`, `credit_contract`, `ValueError`, `_hash`, `list`
- 반환 표현: `{'protocol_id': VERSION, 'runtime_protocol_sha256': protocol['sha256'], 'prompt': PROMPT, 'prompt_sha256': _hash(PROMPT), 'labels': list(LABELS), 'n': 4, 'max_new_tokens': MAX_NEW_TOKEN …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`select_targets` — L65**
```python
select_targets(samples: list[Sample], reread: dict) -> list[dict]
```
- 기존 설명: Use only authored slot values and direct sample ancestry, never outcomes or an audit.
- 직접 호출: `paired_notebook.public_pairs`, `enumerate`, `next`, `ValueError`, `observation_spans`, `values.setdefault(slot, []).append`, `values.setdefault`, `list`, `dict.fromkeys`, `sorted` 외 6개
- 반환 표현: `[{'pair_id': pairs[slot][0], 'channel': channel, **copy.deepcopy(pairs[slot][1]), 'raw_values': varied[slot] if channel == 'NATIVE' else []} for slot, channel in [(slot, 'NATIVE') for s …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`parse_label` — L91**
```python
parse_label(text: str, finish_reason: str) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `text.strip`
- 반환 표현: `label if finish_reason == 'stop' and label in LABELS else None`

**`instruction` — L96**
```python
instruction(samples: list[Sample], schema: str, target: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `episode.public_transcript`, `row.items`, `len`, `PROMPT.format`, `quoted`
- 반환 표현: `PROMPT.format(schema=schema, transcript=quoted(transcript), target=quoted(selected))`

**`collect` — L111**
```python
collect(actors, trees, images, prompts, engine, verifier, record, deadline: float) -> tuple[list[dict], list[dict]]
```
- 기존 설명: Caller owns closed-parent/source/F0/weight lifecycle checks; return rows and skips.
- 직접 호출: `len`, `type`, `prompts.get`, `ValueError`, `min`, `time.monotonic`, `zip`, `skipped.append`, `TimeoutError`, `to_json` 외 32개
- 반환 표현: `(rows, skipped)`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

### D18.115 `fine_tuning/notebook_rl/notebook_pair_to_whole_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_pair_to_whole_probe.py) · SHA256 `0a4a1bb1deedf2410ae3ea26663a4cece71efb87e388dbb5e2d3474939a2bf79`

모듈의 기존 설명: New F0/FT whole-reference inference after verified native pair-action learning.

**`contract` — L27**
```python
contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `restore_protocol`, `protocol.get`, `ValueError`, `reference.contract`, `_hash`, `reference.prompt_contract`
- 반환 표현: `{'protocol_id': VERSION, 'runtime_protocol_sha256': protocol['sha256'], 'reference_contract': reference.contract(), 'prompt_sha256': _hash([reference.prompt_contract(), reference.SCOPE_ …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`historical_parent` — L46**
```python
historical_parent(parent, actors, protocol, model_path, trained) -> unannotated
```
- 기존 설명: Verify old frozen sources/preparation separately from the current CPU approval.
- 직접 호출: `contract`, `closed_parent`, `trained.get`, `read`, `any`, `ValueError`, `native_relationship_update_evidence`, `inventory.items`, `Path(name).is_absolute`, `Path` 외 21개
- 반환 표현: `(trees, evidence)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`baseline_capture` — L101**
```python
baseline_capture(path: Path, protocol: dict, trained: dict, evidence: dict, sources: dict, *, current: dict) -> dict
```
- 기존 설명: Only a newly generated, closed F0 whole-note capture can be the comparator.
- 직접 호출: `read`, `set`, `ValueError`, `json.loads`, `json.dumps`, `checkpoint.digest`, `contract`, `Path`, `len`, `sorted` 외 23개
- 반환 표현: `{'started': started, 'row': row, 'result': result, 'audits': read(path / 'parent-audit.json'), 'files': {str(value): checkpoint.file_hash(value) for value in paths}, 'modes': {str(value …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`collect` — L176**
```python
collect(actors, trees, images, prompts, engine, verifier, record, deadline, *, protocol, arm, record_completion, baseline=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `contract`, `{'baseline': 0, 'trained': 1}.get`, `type`, `len`, `ValueError`, `to_json`, `zip`, `unchanged`, `for_actor`, `episode.solution_samples` 외 41개
- 반환 표현: `(rows, skipped)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.116 `fine_tuning/notebook_rl/notebook_relationship_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_relationship_probe.py) · SHA256 `ed00e40a0d65115c806012b146dd99c68f9a2e0dfedbf5eed009b9e8f4cf1305`

모듈의 기존 설명: Zero-update relationship proposals on unchanged, previously seen native history.

**`_comparisons_valid` — L92**
```python
_comparisons_valid(note, originals: list[Sample]) -> bool
```
- 기존 설명: Validate public coverage and references only; no comparison claim earns verification or credit.
- 직접 호출: `type`, `list`, `any`, `credit._lesson_shape`, `episode.public_transcript`, `len`, `zip`, `set`, `row['comparison'].strip`, `sorted` 외 4개
- 반환 표현: `False`; `sorted(referenced) == list(range(len(note['lessons'])))`

**`contract` — L137**
```python
contract(protocol: dict, *, comparisons: bool=False) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `protocol.get`, `credit.contract`, `ValueError`, `_hash`
- 반환 표현: `{'protocol_id': COMPARISONS_VERSION if comparisons else VERSION, 'runtime_protocol_sha256': protocol['sha256'], 'prompt': prompt, 'native_rules': NATIVE_RULES, 'prompt_sha256': _hash([p …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`collect` — L155**
```python
collect(actors, trees, images, prompts, engine, verifier, record, deadline, *, comparisons: bool=False) -> unannotated
```
- 기존 설명: Caller owns closed-parent/source/F0/weight lifecycle checks; return rows and skips.
- 직접 호출: `len`, `type`, `ValueError`, `zip`, `time.monotonic`, `TimeoutError`, `skipped.append`, `synthesis.audit_episode`, `to_json`, `episode.solution_samples` 외 46개
- 반환 표현: `(rows, skipped)`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

### D18.117 `fine_tuning/notebook_rl/notebook_synthesis.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_synthesis.py) · SHA256 `47fd992d7733acdf8bae17ef9ad64b83247e18e500b48e7bccd1e635fe1ce336`

모듈의 기존 설명: Whole-history structured synthesis reward; never a teacher notebook or routing oracle.

**`public_coverage_contract` — L94**
```python
public_coverage_contract() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'version': PUBLIC_COVERAGE_VERSION, 'inputs': ['text', 'sample_id', 'parent_sample_id'], 'algorithm': 'all attempts in input order; all observation_spans occurrences grouped by canonic …`

**`public_coverage` — L105**
```python
public_coverage(samples: list[Sample]) -> dict
```
- 기존 설명: Index authored claims and lineage using only the three public input fields.
- 직접 호출: `enumerate`, `len`, `ValueError`, `_first_attempt_lines`, `attempts.append`, `pairs.append`, `observation_spans`, `re.fullmatch`, `line['text'].rstrip`, `slots.setdefault(key, []).append` 외 3개
- 반환 표현: `{**public_coverage_contract(), 'attempts': attempts, 'slots': slots, 'raw_observation_lines': raw_lines, 'retry_pairs': pairs}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_sha` — L137**
```python
_sha(text: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(text.encode('utf-8')).hexdigest`, `hashlib.sha256`, `text.encode`
- 반환 표현: `hashlib.sha256(text.encode('utf-8')).hexdigest()`

**`_refs` — L141**
```python
_refs(text: str, spans: list[list[int]]) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `_sha`, `sorted`, `set`, `map`
- 반환 표현: `[{'span': list(pair), 'sha256': _sha(text[pair[0]:pair[1]])} for pair in sorted(set(map(tuple, spans)))]`

**`_output_checks` — L145**
```python
_output_checks(sample: Sample) -> list[dict]
```
- 기존 설명: Inspect final-record presence only, without changing answer extraction or correctness.
- 직접 호출: `extract_final`, `list`, `_FINAL.finditer`, `sample.text.rstrip`, `_first_attempt_lines`, `len`, `sample.text[matches[0].end():].strip`, `matches[0].end`, `matches[0].start`, `_refs`
- 반환 표현: `[]`; `[{'kind': 'OUTPUT', 'rule': 'FINALIZATION', 'locus': 'final_answer', 'status': status, 'identity': ['OUTPUT', 'FINALIZATION', 'same-task'], 'blocked_by': [], 'refs': _refs(sample.text,  …`

**`_identity` — L168**
```python
_identity(trace: dict, rule: str, locus: str) -> Any | None
```
- 기존 설명: Bind cross-attempt operations to authored operands, never array index alone.
- 직접 호출: `trace.get`, `identity`, `re.fullmatch`, `int`, `len`, `sorted`, `event.get`
- 반환 표현: `identity(trace, rule, locus)`; `[kind, rule, 'same-native-task']`; `None`; 외 8개 분기

**`_trace_checks` — L233**
```python
_trace_checks(sample: Sample, report: dict) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_trace_report`, `report.get`, `_sha`, `ValueError`, `_trace`, `_refs`, `type`, `len`, `checks.append`, `_identity` 외 1개
- 반환 표현: `checks`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_observation_checks` — L263**
```python
_observation_checks(sample: Sample, verifier: Any) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `observation_spans`, `grouped.setdefault(key, []).append`, `grouped.setdefault`, `_single_line_scalar`, `sample.text[start:end].rstrip`, `_native_value_shape`, `usable.setdefault(key, []).append`, `usable.setdefault`, `grouped.items`, `usable.get` 외 9개
- 반환 표현: `checks`

**`_native_value_shape` — L296**
```python
_native_value_shape(key: str, value: str) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `slot_id`, `key.startswith`, `native_integer_value`, `re.fullmatch`, `_normalize`
- 반환 표현: `native_integer_value(value)`; `re.fullmatch('[1-9][0-9]?[ \\t]+' + unit, value) is not None`; `_normalize(value, 'relation', key) is not None`; 외 1개 분기

**`_component_unit_mismatch` — L310**
```python
_component_unit_mismatch(key: str, value: str) -> bool
```
- 기존 설명: A recognized wrong dimension is an error; missing units or prose stay unknown.
- 직접 호출: `slot_id`, `re.fullmatch`, `bool`, `key.startswith`
- 반환 표현: `bool(re.fullmatch('comp\\[(?:e[12]|r[1-6])\\]', key) and match and (match[1] != ('V' if key.startswith('comp[e') else 'ohm')))`

**`_single_line_scalar` — L318**
```python
_single_line_scalar(value: Any) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `native_scalar_line`
- 반환 표현: `native_scalar_line(value)`

**`_reinspection_facts` — L324**
```python
_reinspection_facts(note: Any) -> dict[str, str]
```
- 기존 설명: Keep only unambiguous, losslessly parsed native scalar submissions, without private keys.
- 직접 호출: `type`, `note.get`, `any`, `slot_id`, `facts.items`, `canonical.count`, `_single_line_scalar`, `_native_value_shape`, `identity.startswith`, `re.fullmatch` 외 4개
- 반환 표현: `{}`; `result`

**`_reinspection_projection` — L363**
```python
_reinspection_projection(facts: dict[str, str]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `'\n'.join`, `sorted`, `facts.items`
- 반환 표현: `'Observed facts:\n' + '\n'.join((f'{key} = {value}' for key, value in sorted(facts.items()))) + '\nSolution:\n'`

**`_fact_reinspection` — L368**
```python
_fact_reinspection(note: Any, record: dict, problem_id: str, verifier: Any) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_reinspection_facts`, `_reinspection_projection`, `verifier.grade`, `_sha`, `json.loads`, `to_json`, `_hash`
- 반환 표현: `receipt | {'sha256': _hash(receipt)}`

**`_reinspection_feedback` — L379**
```python
_reinspection_feedback(checked: dict, sample: Sample | None) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `checked.get`, `type`, `set`, `_hash`, `receipt.items`, `any`, `_reinspection_facts`, `_sha`, `_reinspection_projection`, `ValueError` 외 5개
- 반환 표현: `[]`; `hints`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_key` — L415**
```python
_key(row: dict) -> tuple[str, str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `(row['kind'], row['rule'], row['locus'])`

**`_merge_checks` — L419**
```python
_merge_checks(rows: list[dict]) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_key`, `list`, `sorted`
- 반환 표현: `[merged[key] for key in sorted(merged)]`

**`_scope_unknown` — L436**
```python
_scope_unknown(row: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `row['kind'] == 'TRACE' and row['status'] == 'UNVERIFIABLE' and ((row['rule'], row['locus']) in _ROOT_UNKNOWNS) and (row['blocked_by'] == [])`

**`_eligible_unknown` — L441**
```python
_eligible_unknown(row: dict, checks: list[dict], *, accounted: bool=False) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row['rule'].startswith`, `supported_unknown`, `_scope_unknown`, `_DEPENDENCIES.get`, `set`, `bool`, `all`, `any`
- 반환 표현: `supported_unknown(row, checks)`; `True`; `bool(row['kind'] == 'TRACE' and row['status'] == 'UNVERIFIABLE' and dependencies and allowed and all((dependency['rule'] in allowed and any((check['kind'] == 'TRACE' and check['status'] …`

**`_unknown_binding` — L462**
```python
_unknown_binding(value: Any, attempt: int, samples: list[Sample], audit: dict) -> dict | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `_claim`, `value.items`, `_selected`, `_key`, `len`, `_eligible_unknown`, `_scope_unknown`, `_sha` 외 1개
- 반환 표현: `None`; `row if empty_scope or _covers(selected, row['refs'] + prerequisite_refs) else None`

**`audit_episode` — L483**
```python
audit_episode(tree: ProblemTree, actor: ActorItem, verifier: Any, *, contract: str=VERSION, evaluation_only: bool=False) -> dict
```
- 기존 설명: Reuse strict raw ancestry/routing checks; audit outcome and issues separately.
- 직접 호출: `ValueError`, `episode_notebook.audit_episode`, `enumerate`, `episode_notebook.solution_samples`, `verifier.trace`, `_merge_checks`, `_observation_checks`, `_trace_checks`, `_scope_unknown`, `_refs` 외 11개
- 반환 표현: `{'version': contract, 'solution_sha256': original['solution_sha256'], 'transcript_sha256': original['transcript_sha256'], 'actor_sha256': _hash(actor.__dict__), 'policy_version': tree.p …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prompt_contract` — L645**
```python
prompt_contract(contract: str=VERSION) -> str
```
- 기존 설명: Canonical public template identity, with item-specific values still unfilled.
- 직접 호출: `ValueError`, `PUBLIC_JSON_SHAPE.replace('{', '{{').replace`, `PUBLIC_JSON_SHAPE.replace`, `prompt.replace`, `prompt.replace('{extension}', extension if contract == BOUNDED_VERSION else '').replace`
- 반환 표현: `prompt.replace('{extension}', extension if contract == BOUNDED_VERSION else '').replace('{attempt_extension}', ', unverified' if contract == BOUNDED_VERSION else '')`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`format_feedback_contract` — L663**
```python
format_feedback_contract() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `_sha`
- 반환 표현: `{'version': FORMAT_DIAGNOSTICS_VERSION, 'max_errors': MAX_FORMAT_ERRORS, 'error_codes': list(FORMAT_ERROR_CODES), 'public_shape_sha256': _sha(PUBLIC_JSON_SHAPE), 'submitted_lesson_count …`

**`format_diagnostics` — L673**
```python
format_diagnostics(text: str) -> dict
```
- 기존 설명: Describe submitted JSON only; never inspect an actor, audit, verifier or line bounds.
- 직접 호출: `json.loads`, `error`, `type`, `note.get`, `len`, `enumerate`, `attempt.get`, `result['empty_attempt_evidence_positions'].append`, `fields`, `facts.values` 외 6개
- 반환 표현: `result`

**`_valid_format_diagnostics` — L810**
```python
_valid_format_diagnostics(value: Any) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `len`, `all`, `sorted`, `re.fullmatch`, `re.findall`
- 반환 표현: `type(value) is dict and set(value) == {'version', 'errors', 'truncated', 'submitted_lesson_count', 'submitted_attempt_count', 'empty_attempt_evidence_positions', 'empty_attempt_evidence …`

**`instruction` — L836**
```python
instruction(samples: list[Sample], audit: dict, observation_instruction: str) -> str
```
- 기존 설명: Quote every raw sibling/retry without private checks, labels or target values.
- 직접 호출: `episode_notebook.public_transcript`, `audit.get`, `_hash`, `ValueError`, `row.items`, `len`, `json.dumps`, `transcript.replace('<', '\\u003c').replace`, `transcript.replace`, `prompt_contract(audit['version']).format` 외 1개
- 반환 표현: `prompt_contract(audit['version']).format(success_attempt=audit['success_attempt'], observation_instruction=observation_instruction, transcript=transcript, retry_pairs=json.dumps([[row[' …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_selected` — L853**
```python
_selected(ranges: Any, sample: Sample, *, allow_empty: bool=False) -> list[dict] | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_first_attempt_lines`, `episode_notebook._evidence_line_ids`, `len`
- 반환 표현: `[]`; `[lines[index - 1] for index in ids] if ids else None`

**`_public_shape_valid` — L861**
```python
_public_shape_valid(sample: Sample, originals: list[Sample]) -> bool
```
- 기존 설명: Check public containers and literal citation bounds, never outcomes or private grades.
- 직접 호출: `format_diagnostics`, `json.loads`, `len`, `enumerate`, `zip`, `any`, `_selected`
- 반환 표현: `False`; `True`

**`select_repair_parent` — L887**
```python
select_repair_parent(candidates: list[Sample], originals: list[Sample], *, policy: str | None) -> dict
```
- 기존 설명: Select the first stopped public-shape-valid sibling; keep every sampled action.
- 직접 호출: `ValueError`, `next`, `enumerate`, `_public_shape_valid`
- 반환 표현: `{'index': selected, 'sample_id': candidates[selected].sample_id, 'reason': 'FIRST_PUBLIC_SHAPE_VALID' if index is not None else 'FALLBACK_FIRST'}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_covers` — L897**
```python
_covers(selected: list[dict], refs: list[dict]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `all`, `any`
- 반환 표현: `bool(refs) and all((any((line['start'] <= ref['span'][0] and ref['span'][1] <= line['end'] for line in selected)) for ref in refs))`

**`_claim` — L902**
```python
_claim(value: Any) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`
- 반환 표현: `type(value) is dict and set(value) == _CLAIM and (value['kind'] in ('OBSERVATION', 'TRACE')) and (type(value['rule']) is str) and (type(value['locus']) is str)`

**`_answer` — L908**
```python
_answer(value: Any, *, nullable: bool=False) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `_single_line_scalar`, `ValueError`, `_validate`
- 반환 표현: `parsed['previous_answer' if nullable else 'corrected_answer']`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_endpoint` — L920**
```python
_endpoint(value: Any, samples: list[Sample], audit: dict, *, issue: bool) -> tuple[dict, list[dict]] | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `len`, `_claim`, `value.items`, `_selected`, `_key`, `_covers`
- 반환 표현: `None`; `(matching[0], selected)`

**`_output_endpoint` — L934**
```python
_output_endpoint(value: Any, samples: list[Sample], audit: dict, *, missing: bool) -> dict | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `audit.get`, `type`, `set`, `len`, `_output_checks`, `actual.get`, `_sha`, `_selected`, `_first_attempt_lines`, `_covers`
- 반환 표현: `None`; `expected[0] if missing or _covers(selected, expected[0]['refs']) else None`

**`_descendant` — L954**
```python
_descendant(earlier: int, later: int, samples: list[Sample]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `parents.get`
- 반환 표현: `True`; `False`

**`_lesson` — L965**
```python
_lesson(value: Any, samples: list[Sample], audit: dict, *, bounded: bool=False) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `_output_endpoint`, `_descendant`, `bool`, `_endpoint`, `len`, `_unknown_binding`, `later.items`, `_scope_unknown` 외 4개
- 반환 표현: `checked`; `{'verified': True, 'claim': value, 'covered_issues': [], 'credit_type': 'OUTPUT_FINALIZATION', 'later_answer_correct': bool(result and result.status == 'CORRECT')}`

**`grade_note` — L1042**
```python
grade_note(text: str, finish_reason: str, tree: ProblemTree, audit: dict, verifier: Any, *, actor: ActorItem, contract: str=VERSION, evaluation_only: bool=False) -> dict
```
- 기존 설명: Freshly re-audit actual evidence; caller-supplied audit claims never earn reward.
- 직접 호출: `audit_episode`, `ValueError`, `_grade_note_audited`
- 반환 표현: `_grade_note_audited(text, finish_reason, tree, fresh, verifier, contract=contract)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_grade_note_audited` — L1051**
```python
_grade_note_audited(text: str, finish_reason: str, tree: ProblemTree, fresh: dict, verifier: Any, *, contract: str=VERSION) -> dict
```
- 기존 설명: Grade against the evidence freshly checked within the calling transaction.
- 직접 호출: `type`, `ValueError`, `episode_notebook.solution_samples`, `_sha`, `_hash`, `format_diagnostics`, `_fact_reinspection`, `json.loads`, `set`, `len` 외 27개
- 반환 표현: `record`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`successful` — L1168**
```python
successful(report: dict, contract: str=VERSION) -> bool
```
- 기존 설명: One explicit success gate for callers; bounded reward never claims full diagnosis.
- 직접 호출: `ValueError`, `set`, `report.get`, `bool`, `type`, `all`, `checks.values`
- 반환 표현: `bool(report.get('reward_version') == contract and type(report.get('reward')) in (int, float) and (report['reward'] == 1) and (type(checks) is dict) and (set(checks) == expected) and all …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`notebook_feedback` — L1185**
```python
notebook_feedback(checked: dict, sample: Sample | None=None) -> str
```
- 기존 설명: Public submitted syntax and rejection booleans/indices; no private expected claims.
- 직접 호출: `checked.get`, `type`, `ValueError`, `_sha`, `set`, `any`, `checks.values`, `enumerate`, `row.get`, `failed_attempts.append` 외 5개
- 반환 표현: `instruction + json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`render_verified` — L1239**
```python
render_verified(report: dict) -> list[str]
```
- 기존 설명: Host display from verified model claims; never a policy target or oracle note.
- 직접 호출: `report.get`, `checked.get`, `rendered.append`
- 반환 표현: `rendered`

### D18.118 `fine_tuning/notebook_rl/notebook_whole_reference_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_whole_reference_probe.py) · SHA256 `7d24344b9af1753855d09ec1f5b337f4e2a86ed58b7314aac7173ecab3101128`

모듈의 기존 설명: Zero-update whole-history reference decisions on the unchanged audited V45 input.

**`contract` — L23**
```python
contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `baseline.contract`, `synthesis.prompt_contract`, `reference.contract`, `_hash`, `reference.prompt_contract`
- 반환 표현: `{'protocol_id': VERSION, 'runtime_protocol_sha256': protocol['sha256'], 'original_prompt': synthesis.prompt_contract(synthesis.BOUNDED_VERSION), 'reference_contract': reference.contract …`

**`_quoted` — L40**
```python
_quoted(text: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps(text, ensure_ascii=False).replace('<', '\\u003c').replace`, `json.dumps(text, ensure_ascii=False).replace`, `json.dumps`
- 반환 표현: `json.dumps(text, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e')`

**`_messages` — L44**
```python
_messages(original, actor, schema, samples, note, success_attempt) -> unannotated
```
- 기존 설명: Keep original image parts and exact quoted reread/drafts; replace only active text.
- 직접 호출: `copy.deepcopy`, `fresh_observation.quoted_context`, `drafts.quoted_context`, `old.count`, `ValueError`, `_quoted`, `p0_body`, `reasoning_trace.instruction_for_question`, `reference.instruction`, `reference.public_inventory`
- 반환 표현: `(messages, [{'before': old, 'after': text}], reference.public_inventory(samples))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`collect` — L64**
```python
collect(actors, trees, images, prompts, engine, verifier, record, deadline, *, record_completion=None) -> unannotated
```
- 기존 설명: The caller owns V45 ancestry, source approvals and the unchanged F0 lifecycle.
- 직접 호출: `len`, `type`, `ValueError`, `to_json`, `zip`, `time.monotonic`, `TimeoutError`, `synthesis.audit_episode`, `skipped.append`, `episode.solution_samples` 외 32개
- 반환 표현: `(rows, skipped)`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

### D18.119 `fine_tuning/notebook_rl/notebook_whole_synthesis_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_whole_synthesis_probe.py) · SHA256 `7d75cb953c6f6d9e2842aa2b792fa74e2fccb6d61698eee7aeda8f3d5ade9990`

모듈의 기존 설명: One zero-update whole-notebook intervention on an unchanged saved draft-assisted input.

**`_ordered_shape` — L71**
```python
_ordered_shape() -> str
```
- 기존 설명: Move the existing public type blocks without changing their fields or placeholders.
- 직접 호출: `synthesis.PUBLIC_JSON_SHAPE.split`, `body.split`
- 반환 표현: `header + '{"corrected_facts":' + facts + ',\n "attempts":' + attempts + ',\n "lessons":' + lessons + ',\n "success_attempt":' + tail`

**`contract` — L81**
```python
contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `protocol.get`, `credit.contract`, `drafts.contract`, `ValueError`, `synthesis.prompt_contract`, `synthesis.public_coverage_contract`, `_hash`, `_ordered_shape`
- 반환 표현: `{'protocol_id': VERSION, 'runtime_protocol_sha256': protocol['sha256'], 'original_prompt': synthesis.prompt_contract(synthesis.BOUNDED_VERSION), 'procedure': PROCEDURE, 'scope_grammar': …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_messages` — L107**
```python
_messages(original: list[dict], actor, schema: str, samples: list[Sample]) -> tuple[list[dict], list[dict], dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `copy.deepcopy`, `p0_body`, `synthesis.public_coverage`, `json.dumps`, `quoted.replace('<', '\\u003c').replace`, `quoted.replace`, `_ordered_shape`, `reasoning_trace.instruction_for_question`, `text.count`, `ValueError` 외 2개
- 반환 표현: `(messages, changes, coverage)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_checked` — L144**
```python
_checked(sample, tree, audit, actor, verifier) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `synthesis.grade_note`, `credit.score_note`, `credit._json_units`, `synthesis.format_diagnostics`, `type`, `set`, `list`, `synthesis._lesson`, `episode.solution_samples`
- 반환 표현: `{'sample_id': sample.sample_id, 'seed': sample.seed, 'text': sample.text, 'finish_reason': sample.finish_reason, 'completion_token_ids': sample.completion_token_ids, 'sampler_logprobs': …`

**`_totals` — L163**
```python
_totals(rows: list[dict]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sum`, `row['score']['fact_rewards'].values`, `row['score']['relationship_rewards'].items`
- 반환 표현: `{'strict_verified_notes': sum((row['strict_check']['reward'] for row in rows)), 'atomic_reward_sum': sum((row['score']['atomic_reward'] for row in rows)), 'verified_fact_units': sum((su …`

**`collect` — L173**
```python
collect(actors, trees, images, prompts, engine, verifier, record, deadline) -> unannotated
```
- 기존 설명: The caller validates historical source/protocol and owns the unchanged F0 weight lifecycle.
- 직접 호출: `len`, `type`, `ValueError`, `to_json`, `zip`, `time.monotonic`, `TimeoutError`, `synthesis.audit_episode`, `skipped.append`, `episode.solution_samples` 외 27개
- 반환 표현: `(rows, skipped)`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

### D18.120 `fine_tuning/notebook_rl/notebook_whole_trained_probe.py`

[현재 소스](../../fine_tuning/notebook_rl/notebook_whole_trained_probe.py) · SHA256 `9907b210c4dda4ed7c4575fa880bd206fb6580104074f53b3a97b778548f32b6`

모듈의 기존 설명: Post-update whole notes on the exact saved pre-update context; never training.

**`contract` — L27**
```python
contract(protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `restore_protocol`, `protocol.get`, `credit.contract`, `drafts.contract`, `ValueError`, `result.update`, `reference.contract`, `_hash`, `pages.contract`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_checked` — L70**
```python
_checked(sample, tree, audit, actor, verifier, *, objective_version=credit.VERSION) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tree.notebook.get('rewarded_notebook', {}).get`, `tree.notebook.get`, `grade`, `credit.score_note`
- 반환 표현: `{'sample_id': sample.sample_id, 'seed': sample.seed, 'text': sample.text, 'finish_reason': sample.finish_reason, 'completion_token_ids': sample.completion_token_ids, 'sampler_logprobs': …`

**`_page_summaries` — L86**
```python
_page_summaries(groups, tree, audit, actor, verifier, scores) -> list[dict]
```
- 기존 설명: Logical reports contain no invented sample or assembled action tokens.
- 직접 호출: `episode.solution_samples`, `range`, `pages.assemble`, `any`, `reference.grade_note`, `summary.update`, `rows.append`
- 반환 표현: `rows`

**`_collect_pages` — L106**
```python
_collect_pages(actor, tree, images, engine, verifier, record, unchanged, audit, policy) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `episode.solution_samples`, `pages.plan`, `note.get`, `_hash`, `type`, `len`, `ValueError`, `enumerate`, `zip`, `seeds` 외 24개
- 반환 표현: `row`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`collect` — L217**
```python
collect(actors, trees, images, prompts, engine, verifier, record, deadline, *, protocol) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `contract`, `len`, `type`, `ValueError`, `to_json`, `zip`, `unchanged`, `synthesis.audit_episode`, `skipped.append`, `for_actor` 외 25개
- 반환 표현: `(rows, skipped)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.121 `fine_tuning/notebook_rl/observations.py`

[현재 소스](../../fine_tuning/notebook_rl/observations.py) · SHA256 `326fd325c972704bffb60b85864ad01c8f20e915a1a23b92aa4121769f980e8b`

모듈의 기존 설명: Parse declared observations only; absent observations are not visual errors.

**`slot_id` — L16**
```python
slot_id(value: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `unicodedata.normalize('NFKC', value).casefold`, `unicodedata.normalize`, `re.sub`, `m[1].replace("'", '').replace`, `m[1].replace`
- 반환 표현: `re.sub('\\[([^\\]]+)\\]', lambda m: '[' + m[1].replace("'", '').replace('"', '') + ']', value)`

**`parse_observations` — L22**
```python
parse_observations(text: str) -> dict[str, list[str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `observation_spans`, `result.setdefault(key, []).append`, `result.setdefault`
- 반환 표현: `result`

**`observation_spans` — L29**
```python
observation_spans(text: str) -> list[tuple[int, int, str, str]]
```
- 기존 설명: Return original assignment offsets only inside accepted observation blocks.
- 직접 호출: `list`, `_BEGIN.finditer`, `enumerate`, `_END.search`, `start.end`, `len`, `end.start`, `starts[i + 1].start`, `text[start.end():end.start()].splitlines`, `_LINE.fullmatch` 외 3개
- 반환 표현: `[]`; `result`

**`normalize_slot` — L49**
```python
normalize_slot(value: str, slot_type: str) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_relation_value`, `unicodedata.normalize('NFKC', value).strip`, `unicodedata.normalize`, `number`, `re.fullmatch`, `str`, `value.replace('♯', '#').replace('♭', 'b').replace`, `value.replace('♯', '#').replace`, `value.replace`, `re.sub` 외 7개
- 반환 표현: `str(parsed) if parsed is not None else None`; `match[1].upper() + match[2] + match[3] if match else None`; `' '.join(value.casefold().split()) or None`; 외 2개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_relation_value` — L81**
```python
_relation_value(value: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `value.strip`, `len`
- 반환 표현: `value[1:-1] if len(value) >= 2 and value[0] in '"\'' and (value[-1] == value[0]) else value`

**`_normalize` — L87**
```python
_normalize(value: str, kind: str, key: str) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `key.startswith`, `unicodedata.normalize('NFKC', _relation_value(value)).strip(' \t').casefold().replace`, `unicodedata.normalize('NFKC', _relation_value(value)).strip(' \t').casefold`, `unicodedata.normalize('NFKC', _relation_value(value)).strip`, `unicodedata.normalize`, `_relation_value`, `re.fullmatch`, `re.sub`, `normalize_slot`
- 반환 표현: `re.sub('[ \\t]+', '', value) if re.fullmatch(pattern, value) else None`; `normalize_slot(value, kind)`

**`native_absent_relations` — L96**
```python
native_absent_relations(keys: Iterable[str], item: VerifierItem) -> tuple[str, ...]
```
- 기존 설명: Prove unregistered relation keys absent from a complete native graph world.
- 직접 호출: `item.accept.get`, `item.problem_id.startswith`, `isinstance`, `world.get`, `_validate_world`, `_facts`, `_slot_types`, `slot_id`, `facts.values`, `tuple` 외 2개
- 반환 표현: `()`; `tuple(sorted({key for key in keys if key not in registered and (match := re.fullmatch(f'{relation}\\[([a-z0-9_]+),([a-z0-9_]+)\\]', key)) and (match[1] in nodes) and (match[2] in ({'l', …`

**`verified_wrong_slots` — L120**
```python
verified_wrong_slots(slots: SlotReport) -> tuple[str, ...]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tuple`, `sorted`, `set`
- 반환 표현: `tuple(sorted(set(slots.wrong_required) | set(slots.false_relations)))`

**`required_observations_verified` — L124**
```python
required_observations_verified(slots: SlotReport | None) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`
- 반환 표현: `bool(slots and slots.required > 0 and (slots.required_correct == slots.required) and (not slots.conflicting) and (not slots.false_relations) and (not slots.unverified_relations))`

**`compare_slots` — L129**
```python
compare_slots(parsed: dict[str, list[str]], item: VerifierItem) -> SlotReport
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `ValueError`, `slot_id`, `len`, `keys.keys`, `parsed.items`, `isinstance`, `all`, `observed.setdefault(key, []).extend`, `observed.setdefault` 외 20개
- 반환 표현: `SlotReport(len(observed), len(required), len(correct & required), tuple(sorted(wrong & required)), len(correct), len(keys), tuple(sorted(conflicts)), tuple(sorted(false_relations)), tup …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.122 `fine_tuning/notebook_rl/pair_action_notebook.py`

[현재 소스](../../fine_tuning/notebook_rl/pair_action_notebook.py) · SHA256 `b55ef68c6ee5e92b0ab63010fb7bcef3d0c3d9d52e1c32181dacee88b4f3c2ed`

모듈의 기존 설명: Fresh independent partial pair actions; raw labels remain the actual sampled actions.

**`prompt_contract` — L46**
```python
prompt_contract() -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `_PROMPT`

**`contract` — L50**
```python
contract() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_hash`, `prompt_contract`
- 반환 표현: `{'version': VERSION, 'prompt_sha256': _hash(prompt_contract()), 'coverage': 'PARTIAL', 'unexamined_coverage': _UNEXAMINED, 'full_notebook_verified': False, 'n': 4, 'max_new_tokens': MAX …`

**`grade_note` — L65**
```python
grade_note(text, finish_reason, tree, audit, verifier, *, actor, contract=synthesis.BOUNDED_VERSION) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `synthesis.grade_note`, `_partial_grade`
- 반환 표현: `_partial_grade(legacy)`

**`_grade_note_audited` — L70**
```python
_grade_note_audited(text, finish_reason, tree, audit, verifier, *, contract=synthesis.BOUNDED_VERSION) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_partial_grade`, `synthesis._grade_note_audited`
- 반환 표현: `_partial_grade(synthesis._grade_note_audited(text, finish_reason, tree, audit, verifier, contract=contract))`

**`_partial_grade` — L74**
```python
_partial_grade(legacy) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `legacy | {'legacy_strict_reward': legacy['reward'], 'notebook_generation_policy': VERSION, 'strict_reward_enabled': False, 'reward': 0.0}`

**`_plan` — L79**
```python
_plan(samples, reread) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `paired_notebook.public_pairs`, `copy.deepcopy`, `inventory.items`
- 반환 표현: `([{'pair_id': pair_id, 'channel': 'OUTPUT' if pair['earlier']['kind'] == 'OUTPUT' else 'NATIVE', **copy.deepcopy(pair), 'raw_values': []} for pair_id, pair in inventory.items()], None i …`

**`_instruction` — L86**
```python
_instruction(samples, schema, target) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.items`, `len`, `episode.public_transcript`, `_PROMPT.format`, `quoted`
- 반환 표현: `_PROMPT.format(schema=schema, transcript=quoted(transcript), target=quoted(selected))`

**`_messages` — L100**
```python
_messages(actor, samples, reread, schema, target) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `p0_body(actor.question, actor.choices).removesuffix`, `p0_body`, `user_message`, `fresh_observation.quoted_context`, `_instruction`, `len`
- 반환 표현: `[user_message(question + '\n\n' + fresh_observation.quoted_context(reread) + _instruction(samples, schema, target), len(actor.image_paths))]`

**`_gid` — L107**
```python
_gid(tree, index) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `f'{tree.problem_id}:v{tree.policy_version}:PAIR_ACTION:target{index}'`

**`_group_hash` — L111**
```python
_group_hash(tree) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_hash`, `json.loads`, `to_json`
- 반환 표현: `_hash([json.loads(to_json(group)) for group in tree.groups if group.kind == 'DIAG'])`

**`validate_update_binding` — L115**
```python
validate_update_binding(tree, *, tokenizer=None, preparation=None, images=None, observation_actor=None, observation_schema=None, observation_master_seed=3407, **other_binding_kwargs) -> None
```
- 기존 설명: Rebuild the public plan and every independent prompt before admitting sampled labels.
- 직접 호출: `to_json`, `tree.notebook.get`, `note.get`, `_hash`, `contract`, `type`, `ValueError`, `other_binding_kwargs.items`, `from_json`, `json.dumps` 외 28개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`bound_target` — L238**
```python
bound_target(tree, sample_id: str, text: str, finish_reason: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_update_binding`, `_bound_target_checked`
- 반환 표현: `_bound_target_checked(tree, sample_id, text, finish_reason)`

**`_bound_target_checked` — L243**
```python
_bound_target_checked(tree, sample_id: str, text: str, finish_reason: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `ValueError`, `copy.deepcopy`
- 반환 표현: `copy.deepcopy(tree.notebook['rewarded_notebook']['target_per_group'][matching[0].group_id]['target'])`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`append` — L251**
```python
append(trees, images, engine, verifier, prompts, cfg, *, actors, on_group=None, on_observation=None, should_stop=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `cfg.validate`, `prompts.get`, `_hash`, `contract`, `prompt_contract`, `set`, `len`, `type`, `any`, `tree.notebook.get` 외 37개
- 반환 표현: `report`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

### D18.123 `fine_tuning/notebook_rl/paired_notebook.py`

[현재 소스](../../fine_tuning/notebook_rl/paired_notebook.py) · SHA256 `3e06bbd15399cd2deed0d3b3eaec89e261c68e196d7aa3d4c73b2bedeba4fae4`

모듈의 기존 설명: Authored ancestor-pair bookkeeping, never verified issues or correction labels.

**`prompt_contract` — L41**
```python
prompt_contract() -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `_PROMPT`

**`_pair_id` — L45**
```python
_pair_id(earlier: dict, later: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `quote`
- 반환 표현: `f"attempt{earlier['attempt']}->attempt{later['attempt']}|{kind}|{rule}|{first}->{second}"`

**`public_pairs` — L51**
```python
public_pairs(samples: list[Sample]) -> dict[str, dict]
```
- 기존 설명: Match public authored identities across actual ancestry, without consulting results.
- 직접 호출: `enumerate`, `len`, `ValueError`, `range`, `reference.public_endpoints(samples).values`, `reference.public_endpoints`, `endpoints[endpoint['attempt']].append`, `_trace`, `set`, `chain.clear` 외 4개
- 반환 표현: `index`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_pair_shape` — L109**
```python
_pair_shape(value) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `bool`
- 반환 표현: `type(value) is dict and set(value) == {'pair_id', 'relation'} and (type(value['pair_id']) is str) and bool(value['pair_id']) and (type(value['relation']) is str) and (value['relation']  …`

**`expand_lesson` — L115**
```python
expand_lesson(value, index: dict[str, dict]) -> dict | None
```
- 기존 설명: Copy a chosen pair only; semantic verification remains with the existing lesson checker.
- 직접 호출: `_pair_shape`, `type`, `index.get`, `set`, `any`, `_range_shape`, `_pair_id`, `copy.deepcopy`
- 반환 표현: `None`; `{'rule': earlier['rule'], 'earlier': copy.deepcopy(earlier), 'later': copy.deepcopy(later), 'relation': value['relation']}`

**`instruction` — L138**
```python
instruction(samples: list[Sample], audit: dict, schema: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `episode.public_transcript`, `audit.get`, `_hash`, `ValueError`, `row.items`, `len`, `public_pairs(samples).items`, `public_pairs`, `prompt_contract().format`, `prompt_contract` 외 1개
- 반환 표현: `prompt_contract().format(schema=schema, pairs=quoted(pairs), transcript=quoted(rows))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`classify` — L158**
```python
classify(text: str, finish_reason: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_json_units`, `set`, `type`, `any`, `_pair_shape`
- 반환 표현: `'TRUNCATED_PARTIAL_NOTEBOOK'`; `'INVALID_PARTIAL_NOTEBOOK'`; `'PARTIAL_NOTEBOOK_CAPTURED' if note['lessons'] else 'NO_NOTEBOOK_CLAIMS'`

### D18.124 `fine_tuning/notebook_rl/partial_notebook.py`

[현재 소스](../../fine_tuning/notebook_rl/partial_notebook.py) · SHA256 `1dae742c8c5fb303cd84663cc996490abf65250cc1e0633bc8960c3179a5d745`

모듈의 기존 설명: Partial model-written notebook input and public output shape, never truth or learning.

**`prompt_contract` — L45**
```python
prompt_contract() -> str
```
- 기존 설명: Canonical public template; no item-specific facts or teacher examples.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `_PROMPT`

**`instruction` — L50**
```python
instruction(samples: list[Sample], audit: dict, schema: str) -> str
```
- 기존 설명: Bind the entire public transcript; private audit contents never enter the prompt.
- 직접 호출: `episode.public_transcript`, `audit.get`, `_hash`, `ValueError`, `row.items`, `len`, `json.dumps`, `prompt_contract().format`, `prompt_contract`, `transcript.replace('<', '\\u003c').replace` 외 1개
- 반환 표현: `prompt_contract().format(native_rules=_NATIVE_RULES, schema=schema, pairs=json.dumps([[row['parent'], row['attempt']] for row in public if row['parent'] is not None]), transcript=transc …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`classify` — L66**
```python
classify(text: str, finish_reason: str) -> str
```
- 기존 설명: Classify public JSON shape and claim presence, without checking claim truth or rewards.
- 직접 호출: `credit._json_units`, `set`, `type`, `any`, `synthesis._single_line_scalar`, `synthesis._native_value_shape`, `note['corrected_facts'].items`, `credit._lesson_shape`
- 반환 표현: `'TRUNCATED_PARTIAL_NOTEBOOK'`; `'INVALID_PARTIAL_NOTEBOOK'`; `'PARTIAL_NOTEBOOK_CAPTURED' if note['corrected_facts'] or note['lessons'] else 'NO_NOTEBOOK_CLAIMS'`

### D18.125 `fine_tuning/notebook_rl/phase_a.py`

[현재 소스](../../fine_tuning/notebook_rl/phase_a.py) · SHA256 `dac7f3ff60a10e99a421c155995e5840345513bd68153e5202ba64164642b2e6`

모듈의 기존 설명: Fixed-F0 train measurements, never Phase-A or training approval.

**`_snapshot` — L34**
```python
_snapshot(value) -> unannotated
```
- 기존 설명: Keep malformed returned payloads durable, including nonfinite numbers.
- 직접 호출: `is_dataclass`, `isinstance`, `_snapshot`, `asdict`, `type`, `str`, `value.items`, `math.isfinite`, `repr`
- 반환 표현: `_snapshot(asdict(value))`; `{str(k): _snapshot(v) for k, v in value.items()}`; `[_snapshot(v) for v in value]`; 외 3개 분기

**`_cap` — L49**
```python
_cap(lengths: list[int]) -> tuple[int | None, int | None]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `math.ceil`, `len`, `min`, `max`
- 반환 표현: `(None, None)`; `(min(4096, max(2048, math.ceil(p95 / 256) * 256)), p95)`

**`_turn_cap` — L56**
```python
_turn_cap(observations: list[dict], actors, kind: str, blocked: list[dict], upstream=()) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_cap`, `len`, `sorted`, `sum`, `status`, `bool`
- 반환 표현: `{'status': status(len(rows), len(complete), len(blocked), len(upstream)), 'n': len(rows), 'completed_final_n': len(complete), 'cap': cap, 'complete_final_p95': p95, 'method': 'nearest-r …`

**`_attribution` — L96**
```python
_attribution(observations: list[dict], actors, kind: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sum`, `len`
- 반환 표현: `{'n_obs_candidate_items': len(eligible), 'n_obs_observed_items': len(observed_ids), 'n_obs_observations': len(rows), 'failed_observations': len(failed), 'cause_counts': {cause: sum((row …`

**`measure_candidates` — L125**
```python
measure_candidates(build_root: Path, output: Path, *, engine: Any, protocol: dict, input_binding: dict, deadline_timestamp: float, seed: int=3407, forbidden_roots=(), clock: Callable[[], float]=time.time) -> dict
```
- 기존 설명: Measure all 576 train candidates and a preselected 128-item retry subset.
- 직접 호출: `_measure_candidates`
- 반환 표현: `_measure_candidates(build_root, output, engine=engine, protocol=protocol, input_binding=input_binding, deadline_timestamp=deadline_timestamp, seed=seed, forbidden_roots=forbidden_roots, …`

**`_measure_candidates` — L141**
```python
_measure_candidates(build_root: Path, output: Path, *, engine: Any, protocol: dict, input_binding: dict, deadline_timestamp: float, seed: int=3407, forbidden_roots=(), clock: Callable[[], float]=time.time, expected_candidates: int, retry_subset_size: int) -> dict
```
- 기존 설명: Bounded authored-fixture harness; production entry always supplies 576/128.
- 직접 호출: `_frozen`, `type`, `math.isfinite`, `any`, `ValueError`, `getattr`, `callable`, `_safe_root`, `Path`, `Path(__file__).resolve` 외 55개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`, `interrupted`; 전파되는 callee 예외는 별도.

### D18.126 `fine_tuning/notebook_rl/policy.py`

[현재 소스](../../fine_tuning/notebook_rl/policy.py) · SHA256 `45cdcf0563fe665b658d01deca4caff1018933d0944cdc9c81dc9a3777cb472d`

모듈의 기존 설명: Selective policy gradients with an immutable adapter-disabled F0 reference.

**`enable_deterministic_execution` — L23**
```python
enable_deterministic_execution() -> dict
```
- 기존 설명: Require strict CUDA reproducibility, never relax gradient comparisons.
- 직접 호출: `os.environ.get`, `torch.cuda.is_initialized`, `ValueError`, `torch.use_deterministic_algorithms`, `torch.are_deterministic_algorithms_enabled`, `torch.is_deterministic_algorithms_warn_only_enabled`
- 반환 표현: `{'algorithms_enabled': torch.are_deterministic_algorithms_enabled(), 'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(), 'cublas_workspace_config': os.environ['CUBLAS_WO …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `VisionRoute` — L42**

기저: `object`.

**`VisionRoute.__init__` — L43**
```python
VisionRoute.__init__(self, vision_lora_params: list[nn.Parameter]) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`VisionRoute.apply` — L47**
```python
VisionRoute.apply(self, vision_on: bool) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `parameter.requires_grad_`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `PolicyBundle` — L56**

기저: `object`.

```python
model: Any
processor: Any
inventory: dict
params: dict[str, list[nn.Parameter]]
route: VisionRoute
base_sha256: dict[str, str]
```

**`_canonical_frozen` — L65**
```python
_canonical_frozen(bundle: PolicyBundle) -> Iterator[tuple[str, torch.Tensor]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bundle.model.get_base_model().named_parameters`, `bundle.model.get_base_model`, `name.replace('.base_layer.', '.').replace`, `name.replace`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`configure_policy` — L72**
```python
configure_policy(base: Any, processor: Any, arm: str, protocol: dict, *, enforce_counts: bool=True) -> PolicyBundle
```
- 기존 설명: Also used with a tiny real Qwen architecture in CPU tests, never tiny weights in GPU runs.
- 직접 호출: `hasattr`, `ValueError`, `inspect_modules`, `len`, `set`, `tensor_sha256`, `base.named_parameters`, `base.requires_grad_`, `get_peft_model`, `LoraConfig` 외 26개
- 반환 표현: `bundle`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`build_policy` — L127**
```python
build_policy(model_path: Path, arm: str, protocol: dict) -> PolicyBundle
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `load_base`, `AutoProcessor.from_pretrained`, `str`, `configure_policy`
- 반환 표현: `configure_policy(base, processor, arm, protocol)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_logprobs` — L142**
```python
_logprobs(bundle: PolicyBundle, sample_inputs: dict, completion_len: int, chunk: int) -> torch.Tensor
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `sample_inputs.keys`, `ValueError`, `isinstance`, `bool`, `(mask == 1).all`, `type`, `bundle.model.get_base_model`, `next`, `base.parameters` 외 13개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`completion_logprobs` — L182**
```python
completion_logprobs(bundle: PolicyBundle, sample_inputs: dict, completion_len: int, *, chunk: int=1024) -> torch.Tensor
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bundle.route.apply`, `_logprobs`
- 반환 표현: `_logprobs(bundle, sample_inputs, completion_len, chunk)`

**`reference_logprobs` — L187**
```python
reference_logprobs(bundle: PolicyBundle, sample_inputs: dict, completion_len: int, *, chunk: int=1024) -> torch.Tensor
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.no_grad`, `bundle.model.disable_adapter`, `_logprobs`, `bundle.route.apply`
- 반환 표현: `_logprobs(bundle, sample_inputs, completion_len, chunk)`

**`assert_base_unchanged` — L197**
```python
assert_base_unchanged(bundle: PolicyBundle) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tensor_sha256`, `_canonical_frozen`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`merged_named_tensors` — L203**
```python
merged_named_tensors(bundle: PolicyBundle) -> Iterator[tuple[str, torch.Tensor]]
```
- 기존 설명: Independent BF16 tensors; never mutate PEFT/base or reuse an in-place merge.
- 직접 호출: `bundle.model.get_base_model`, `set`, `base.named_parameters`, `name.replace('.base_layer.', '.').replace`, `name.replace`, `ValueError`, `seen.add`, `torch.no_grad`, `parameter.detach().float`, `parameter.detach` 외 6개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.127 `fine_tuning/notebook_rl/prompts.py`

[현재 소스](../../fine_tuning/notebook_rl/prompts.py) · SHA256 `f7783b08416e46bb3d9a39aa1fb056d91a0efc95adf95eef41f92ea050f07687`

모듈의 기존 설명: Gold-free prompt builders; retry histories are joined by token ID elsewhere.

**`native_solution_prompt_contract` — L37**
```python
native_solution_prompt_contract() -> dict
```
- 기존 설명: Freeze public task instructions, without generating a world or a reference answer.
- 직접 호출: `graph._question`, `questions.append`, `instruction_for_question`
- 반환 표현: `{'trace_contract': VERSION, 'turn1': NATIVE_TURN1, 'retry': NATIVE_RETRY, 'instructions': [{'source': source, 'question': question, 'instruction': instruction_for_question(question, sou …`

**`circuit_solution_prompt_contract` — L52**
```python
circuit_solution_prompt_contract() -> dict
```
- 기존 설명: Bind every heldout ladder query's public grammar without generating a solution.
- 직접 호출: `circuit._question`, `instruction_for_question`
- 반환 표현: `{'trace_contract': circuit_trace.VERSION, 'turn1': NATIVE_TURN1, 'retry': NATIVE_RETRY, 'scope': circuit_trace.SCOPE, 'source': circuit.GENERATOR_VERSION, 'observation_instruction': cir …`

**클래스 `_UniqueLoader` — L68**

기저: `yaml.SafeLoader`.

**`_mapping` — L72**
```python
_mapping(loader, node) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `loader.construct_object`, `isinstance`, `ValueError`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_prompts` — L85**
```python
load_prompts(path: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.read_bytes`, `yaml.load`, `type`, `set`, `ValueError`, `_TOKENS.items`, `isinstance`, `data[key].strip`, `_PLACEHOLDER.findall`, `any` 외 4개
- 반환 표현: `data | {'sha256': hashlib.sha256(raw).hexdigest()}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_render` — L102**
```python
_render(template: str, **values: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_PLACEHOLDER.sub`
- 반환 표현: `_PLACEHOLDER.sub(lambda m: values[m[1]], template)`

**`p0_body` — L107**
```python
p0_body(question: str, choices: tuple[str, ...]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `question.strip`, `ValueError`, `len`, `any`, `v.strip`, `''.join`, `chr`, `enumerate`
- 반환 표현: `body`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_observation` — L119**
```python
_observation(mode: Mode, family: str, prompts: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `''`; `prompts['observation_instruction'][family]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`for_actor` — L129**
```python
for_actor(actor: ActorItem, prompts: dict) -> dict
```
- 기존 설명: Select a protocol-bound public source schema without reading verifier slots.
- 직접 호출: `ValueError`, `instruction_for_question`, `prompts.get`, `type`, `overrides.get`, `instruction.strip`
- 반환 표현: `prompts`; `prompts | {'observation_instruction': prompts['observation_instruction'] | {actor.family: instruction}}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`turn1_text` — L157**
```python
turn1_text(actor: ActorItem, prompts: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_actor`, `for_actor`, `_render`, `p0_body`, `_observation`
- 반환 표현: `_render(NATIVE_TURN1, P0_BODY=p0_body(actor.question, actor.choices), TRACE_INSTRUCTION=prompts['_native_trace_instruction'], OBSERVATION_INSTRUCTION=_observation(actor.mode, actor.fami …`; `_render(prompts['turn1_' + actor.mode], P0_BODY=p0_body(actor.question, actor.choices), OBSERVATION_INSTRUCTION=_observation(actor.mode, actor.family, prompts))`

**`retry_text` — L169**
```python
retry_text(status: Status, prev_answer: str | None, mode: Mode, family: str, prompts: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `_observation`, `prompts.get`, `isinstance`, `instruction.strip`, `prev_answer.strip`, `_render`
- 반환 표현: `_render(NATIVE_RETRY, FEEDBACK=feedback, TRACE_INSTRUCTION=instruction, OBSERVATION_INSTRUCTION=obs)`; `_render(prompts['retry_incorrect'], PREV_ANSWER=prev_answer, OBS_SECTION=section)`; `_render(prompts['retry_invalid'], REASON=reasons[status], OBS_SECTION=section)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`diagnose_text` — L208**
```python
diagnose_text(k: int, prompts: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `_render`, `str`
- 반환 표현: `_render(prompts['diagnose'], K=str(k), K_MINUS_1=str(k - 1))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`oracle_text` — L214**
```python
oracle_text(actor: ActorItem, fact_lines: tuple[str, ...], prompts: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_actor`, `any`, `isinstance`, `line.strip`, `ValueError`, `_render`, `'\n'.join`, `p0_body`
- 반환 표현: `_render(prompts['oracle_facts'], FACT_LINES='\n'.join(fact_lines), P0_BODY=p0_body(actor.question, actor.choices))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`probe_text` — L222**
```python
probe_text(observed_text: str, actor: ActorItem, prompts: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_actor`, `isinstance`, `observed_text.strip`, `ValueError`, `_render`, `p0_body`
- 반환 표현: `_render(prompts['self_perception_probe'], OBSERVED_TEXT=observed_text, P0_BODY=p0_body(actor.question, actor.choices))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`user_message` — L230**
```python
user_message(text: str, n_images: int) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `type`, `ValueError`, `range`
- 반환 표현: `{'role': 'user', 'content': [*({'type': 'image'} for _ in range(n_images)), {'type': 'text', 'text': text}]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.128 `fine_tuning/notebook_rl/quantized_cache.py`

[현재 소스](../../fine_tuning/notebook_rl/quantized_cache.py) · SHA256 `3a4541708c94786795eb9299ffc6645f8294fdfa5b41eb0aa75a66a1b1fbc665`

모듈의 기존 설명: Offline NF4 cache readiness: real generation/reload, no optimizer or learning.

**`_sources` — L20**
```python
_sources() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(__file__).resolve`, `Path`, `paths.append`, `str`, `path.relative_to`, `file_hash`
- 반환 표현: `{str(path.relative_to(here.parent)): file_hash(path) for path in paths}`

**`_requests` — L29**
```python
_requests(output: Path) -> list
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw(red).rectangle`, `ImageDraw.Draw`, `ImageDraw.Draw(blue).ellipse`, `image.save`, `enumerate`, `requests.append`, `RolloutRequest`
- 반환 표현: `requests`

**`_measure` — L49**
```python
_measure(bundle, protocol: dict, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `SharedPolicyRolloutEngine`, `_requests`, `bundle.processor.tokenizer.encode`, `synchronize`, `SyncProbe`, `engine.sample`, `engine.probe_logprobs`, `zip`, `assert_quantized_base_unchanged`, `asdict` 외 3개
- 반환 표현: `{'sync': sync, 'generations': [asdict(result) for result in generated], 'completion_logprobs': probabilities, 'images': {p.name: file_hash(p) for p in sorted(output.glob('*.png'))}}`

**`_compare` — L68**
```python
_compare(before: dict, after: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `ValueError`, `zip`, `left.items`, `right.items`, `deltas.extend`, `abs`, `any`, `math.isfinite`, `max`
- 반환 표현: `{'generations': len(after['generations']), 'probabilities': len(deltas), 'max_abs_logprob_delta': max(deltas), 'tolerance': 1e-06, 'token_ids_identical': True}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`block_original_access` — L95**
```python
block_original_access(original_root: Path) -> dict
```
- 기존 설명: Process-local proof that a cache reload cannot read its original HF model.
- 직접 호출: `original_root.resolve`, `original_root.is_absolute`, `len`, `ValueError`, `sys.addaudithook`
- 반환 표현: `attempts`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L116**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_mutually_exclusive_group`, `generation.add_argument`, `reference.add_argument`, `objective.add_argument`, `parser.parse_args`, `parser.error`, `args.output.mkdir`, `time.monotonic` 외 40개
- 반환 표현: `code`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.129 `fine_tuning/notebook_rl/quantized_policy.py`

[현재 소스](../../fine_tuning/notebook_rl/quantized_policy.py) · SHA256 `387ec3e316f293cbd0cd0a21138180a50056a762b3ca4b92ad8f1b642dc180d9`

모듈의 기존 설명: Explicit local NF4 policy; the existing BF16 protocol and loader stay separate.

**`quantization_contract` — L35**
```python
quantization_contract() -> dict
```
- 기존 설명: A new protocol must explicitly bind this format, never inherit BF16 identity.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'method': 'bitsandbytes', 'load_in_4bit': True, 'quant_type': 'nf4', 'compute_dtype': 'bfloat16', 'use_double_quant': True, 'storage_dtype': 'uint8', 'linear_input_dtype': 'bfloat16',  …`

**`_versions` — L43**
```python
_versions() -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `version`, `ValueError`
- 반환 표현: `actual`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_protocol` — L50**
```python
_validate_protocol(model_path: Path, arm: str, protocol: dict, device_index: int, *, require_snapshot: bool=True) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `model_path.is_dir`, `ValueError`, `protocol.get('initial_checkpoint', {}).get`, `protocol.get`, `protocol['initial_checkpoint'].get`, `quantization_contract`, `protocol.get('model', {}).get`, `list`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_inspect_quantized_base` — L64**
```python
_inspect_quantized_base(base: Any, device: torch.device) -> list[str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `inspect_modules`, `getattr`, `len`, `set`, `any`, `base.parameters`, `ValueError`, `base.named_modules`, `isinstance`, `name.startswith` 외 1개
- 반환 표현: `quantized`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_cuda_device` — L94**
```python
_cuda_device(device_index: int) -> torch.device
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.cuda.is_available`, `torch.cuda.device_count`, `ValueError`, `torch.cuda.device`, `torch.cuda.is_bf16_supported`, `torch.device`
- 반환 표현: `torch.device('cuda', device_index)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_bf16_linear_input` — L103**
```python
_bf16_linear_input(module: nn.Module, inputs: tuple) -> tuple
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `inputs[0].is_floating_point`, `ValueError`, `inputs[0].to`
- 반환 표현: `inputs if inputs[0].dtype == torch.bfloat16 else (inputs[0].to(torch.bfloat16), *inputs[1:])`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_register_bf16_inputs` — L110**
```python
_register_bf16_inputs(base: Any) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `base.modules`, `isinstance`, `module.register_forward_pre_hook`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`quantization_state_sha256` — L118**
```python
quantization_state_sha256(model: Any) -> dict
```
- 기존 설명: Hash every scale/code and packed shape/dtype/blocksize, beyond weight bytes.
- 직접 호출: `hasattr`, `model.get_base_model`, `base.named_modules`, `isinstance`, `name.removesuffix`, `getattr`, `ValueError`, `state.as_dict`, `set`, `any` 외 8개
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`assert_quantized_base_unchanged` — L145**
```python
assert_quantized_base_unchanged(bundle: PolicyBundle) -> None
```
- 기존 설명: Call around updates/checkpoints/reloads in addition to ordinary BF16 checks.
- 직접 호출: `assert_base_unchanged`, `bundle.inventory.get`, `provenance.get`, `quantization_contract`, `_versions`, `quantization_state_sha256`, `ValueError`, `bundle.model.modules`, `isinstance`, `module._forward_pre_hooks.values`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`build_quantized_policy` — L160**
```python
build_quantized_policy(model_path: Path, arm: str, protocol: dict, *, device_index: int=0) -> PolicyBundle
```
- 기존 설명: Load original local BF16 source into NF4 on one CUDA device, without offload.
- 직접 호출: `Path`, `_validate_protocol`, `_build_quantized_policy`
- 반환 표현: `_build_quantized_policy(model_path, arm, protocol, device_index)`

**`_build_quantized_policy` — L172**
```python
_build_quantized_policy(model_path: Path, arm: str, protocol: dict, device_index: int) -> PolicyBundle
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_versions`, `_cuda_device`, `BitsAndBytesConfig`, `Qwen3VLForConditionalGeneration.from_pretrained`, `str`, `_inspect_quantized_base`, `quantization_state_sha256`, `_register_bf16_inputs`, `AutoProcessor.from_pretrained`, `configure_policy` 외 6개
- 반환 표현: `bundle`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_frozen_state_dict` — L203**
```python
_frozen_state_dict(bundle: PolicyBundle) -> dict[str, torch.Tensor]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bundle.model.get_base_model().state_dict().items`, `bundle.model.get_base_model().state_dict`, `bundle.model.get_base_model`, `name.replace('.base_layer.', '.').replace`, `name.replace`, `ValueError`, `set`, `state.keys`, `bundle.inventory['quantization']['state_sha256'].items`, `components.items` 외 4개
- 반환 표현: `state`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`save_quantized_base` — L223**
```python
save_quantized_base(bundle: PolicyBundle, destination: Path) -> dict
```
- 기존 설명: Save immutable NF4 F0 and processor, including original (untrained) Mergers.
- 직접 호출: `assert_quantized_base_unchanged`, `ValueError`, `_frozen_state_dict`, `Path`, `destination.mkdir`, `bundle.model.get_base_model`, `save_torch_state_dict`, `getattr`, `_serialized_state_sha256`, `set` 외 17개
- 반환 표현: `metadata`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_serialized_state_sha256` — L266**
```python
_serialized_state_sha256(cache: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `cache.glob`, `safe_open`, `str`, `stream.keys`, `ValueError`, `stream.get_tensor`, `tensor_sha256`, `list`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`read_quantized_base_metadata` — L282**
```python
read_quantized_base_metadata(cache: Path) -> dict
```
- 기존 설명: Reject partial exports, foreign identity, path tricks and changed payloads.
- 직접 호출: `Path`, `cache.is_dir`, `cache.is_symlink`, `ValueError`, `list`, `cache.iterdir`, `any`, `p.is_file`, `p.is_symlink`, `(cache / _CACHE_METADATA).is_file` 외 13개
- 반환 표현: `metadata`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_quantized_policy` — L311**
```python
load_quantized_policy(cache: Path, arm: str, protocol: dict, *, device_index: int=0) -> PolicyBundle
```
- 기존 설명: Rebuild fresh NF4 F0 from a verified cache, before restoring RL trainables.
- 직접 호출: `Path`, `read_quantized_base_metadata`, `_validate_protocol`, `_build_quantized_policy`, `ValueError`, `assert_quantized_base_unchanged`
- 반환 표현: `bundle`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.130 `fine_tuning/notebook_rl/reasoning_trace.py`

[현재 소스](../../fine_tuning/notebook_rl/reasoning_trace.py) · SHA256 `c3f1ce541bd545c5f5a46d6f828a46c34e44c53bbbcf1eb93e9c985af101d6e2`

모듈의 기존 설명: Private checks of model-authored native traces; never supplies a reference trace.

**`_sha` — L27**
```python
_sha(text: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(text.encode('utf-8')).hexdigest`, `hashlib.sha256`, `text.encode`
- 반환 표현: `hashlib.sha256(text.encode('utf-8')).hexdigest()`

**`_task` — L31**
```python
_task(question: str, source: str) -> tuple[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `graph._question`, `ValueError`
- 반환 표현: `('shortest', '')`; `(kind, traversal if kind == 'tree' else '')`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`instruction_for_question` — L44**
```python
instruction_for_question(question: str, source: str) -> str
```
- 기존 설명: Public output shape and task rules only; no generated values or teacher trace.
- 직접 호출: `circuit_instruction`, `_task`
- 반환 표현: `circuit_instruction(question, source)`; `f'Trace contract {VERSION}: make the FIRST nonempty line after Solution: exactly Trace: followed by one JSON object on the same line. Assert your own executed trace, not a quotation, ex …`

**`_contract` — L81**
```python
_contract(actor: ActorItem, item: VerifierItem) -> tuple[dict, str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_task`, `len`, `json.dumps`, `ValueError`, `graph._validate_world`, `enumerate`, `str`, `graph._facts`, `key.startswith`, `graph._slot_types` 외 2개
- 반환 표현: `(world, kind, traversal)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_object` — L109**
```python
_object(pairs: list[tuple[str, Any]]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_reject_constant` — L118**
```python
_reject_constant(value: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_trace` — L122**
```python
_trace(text: str) -> tuple[dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `text.count`, `ValueError`, `list`, `re.finditer`, `headers[0].end`, `re.match`, `re.search`, `match.start`, `match.end` 외 3개
- 반환 표현: `(payload, {'line': line, 'span': [start, end], 'sha256': _sha(line)})`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_labels` — L145**
```python
_labels(value: Any) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `len`, `all`, `re.fullmatch`
- 반환 표현: `type(value) is list and len(value) <= 32 and all((type(v) is str and re.fullmatch('[A-Za-z0-9_]{1,16}', v) for v in value))`

**`native_integer_value` — L150**
```python
native_integer_value(value: object) -> bool
```
- 기존 설명: Native dimensionless syntax: 0 or nonzero-led ASCII digits (<=32), outer space/tab only.
- 직접 호출: `type`, `re.fullmatch`
- 반환 표현: `type(value) is str and re.fullmatch('[ \\t]*(?:0|[1-9][0-9]{0,31})[ \\t]*', value) is not None`

**`native_scalar_line` — L159**
```python
native_scalar_line(value: object) -> bool
```
- 기존 설명: A nonblank printable/tab-only scalar with no splitlines separators; no rewriting.
- 직접 호출: `type`, `bool`, `value.strip`, `value.splitlines`, `all`, `char.isprintable`
- 반환 표현: `type(value) is str and bool(value.strip(' \t')) and (value.splitlines() == [value]) and all((char == '\t' or char.isprintable() for char in value))`

**`_response_scope` — L165**
```python
_response_scope(text: str, trace_end: int) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `native_scalar_line`, `text.replace('\r\n', '').replace`, `text.replace`, `re.search`, `header.start`, `re.match`, `facts_header.end`, `reversed`, `observation_spans`, `key.startswith` 외 4개
- 반환 표현: `False`; `not prefix.strip() and re.fullmatch('Final answer: [^\\r\\n]+', text[trace_end:].strip()) is not None`

**`_schema` — L185**
```python
_schema(trace: dict, kind: str, traversal: str, pair: bool) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `trace.get`, `set`, `_labels`, `len`, `type`, `all`, `re.fullmatch`, `bool`
- 반환 표현: `False`; `set(trace) == {'kind', 'nodes', 'edge_ids', 'weights', 'cost'} and _labels(trace['nodes']) and (2 <= len(trace['nodes']) <= 32) and (set(trace['nodes']) <= set('ABCDEFGH')) and (trace[' …`; `set(trace) == {'kind', 'steps'} and type(trace['steps']) is list and (1 <= len(trace['steps']) <= 32) and all((type(step) is dict and set(step) == {'node', 'new'} and _labels([step['nod …`; 외 2개 분기

**`check_trace` — L215**
```python
check_trace(*, actor: ActorItem, item: VerifierItem, text: str, finish_reason: str) -> dict
```
- 기존 설명: Gold-independent audit. A true final answer never erases a false trace operation.
- 직접 호출: `check_circuit`, `_sha`, `len`, `observation_spans`, `_contract`, `add`, `_finish`, `_trace`, `_schema`, `_response_scope` 외 18개
- 반환 표현: `check_circuit(actor=actor, item=item, text=text, finish_reason=finish_reason)`; `_finish(report)`

**`_finish` — L359**
```python
_finish(report: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `sorted`
- 반환 표현: `report`

**`_inventory` — L370**
```python
_inventory(trace: dict) -> list[tuple[str, str]]
```
- 기존 설명: Derive exact public regions from authored JSON, without solving the private world.
- 직접 호출: `trace.get`, `inventory`, `rows.append`, `range`, `len`, `rows.extend`, `enumerate`, `min`
- 반환 표현: `inventory(trace)`; `rows`

**`_dependencies` — L400**
```python
_dependencies(rows: list[dict], trace: dict | None) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `type`, `set`, `earlier.get`, `re.fullmatch`, `int`, `trace.get`, `len`, `failed_path.append`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_trace_report` — L449**
```python
validate_trace_report(report: dict, text: str, finish_reason: str) -> None
```
- 기존 설명: Validate bounded JSON and exact original evidence binding across a process boundary.
- 직접 호출: `report.get`, `validate_report`, `len`, `json.dumps`, `set`, `_sha`, `type`, `ValueError`, `_trace`, `any` 외 11개
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.131 `fine_tuning/notebook_rl/referenced_notebook.py`

[현재 소스](../../fine_tuning/notebook_rl/referenced_notebook.py) · SHA256 `bcc903aa2e21647fe7eea19b15917b0f5179dc7338e0cc498d0e311c3a8c8a75`

모듈의 기존 설명: Text-derived endpoint bookkeeping; no reference carries a private correctness judgment.

**`prompt_contract` — L43**
```python
prompt_contract() -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `_PROMPT`

**`public_endpoints` — L47**
```python
public_endpoints(samples: list[Sample]) -> dict[str, dict]
```
- 기존 설명: Index authored text and ordinal attempts only, including unsupported or incorrect claims.
- 직접 호출: `enumerate`, `observation_spans`, `text.splitlines`, `re.fullmatch`, `json.loads`, `{'shortest': 'weights', 'bfs': 'steps', 'tree': 'events', 'dfa': 'states'}.get`, `trace.get`, `type`, `rows.extend`, `_inventory` 외 3개
- 반환 표현: `index`

**`_reference_shape` — L76**
```python
_reference_shape(value) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `all`, `bool`
- 반환 표현: `type(value) is dict and set(value) == {'relation', 'earlier_ref', 'later_ref'} and (type(value['relation']) is str) and (value['relation'] in _RELATIONS) and all((type(value[key]) is st …`

**`expand_lesson` — L82**
```python
expand_lesson(value, index: dict[str, dict]) -> dict | None
```
- 기존 설명: Copy model-chosen bookkeeping only; semantic, channel and ancestry checks remain external.
- 직접 호출: `_reference_shape`, `type`, `index.get`, `earlier.get`, `copy.deepcopy`
- 반환 표현: `None`; `{'rule': earlier['rule'], 'earlier': copy.deepcopy(earlier), 'later': copy.deepcopy(later), 'relation': value['relation']}`

**`instruction` — L93**
```python
instruction(samples: list[Sample], audit: dict, schema: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `episode.public_transcript`, `audit.get`, `_hash`, `ValueError`, `row.items`, `len`, `public_endpoints(samples).items`, `public_endpoints`, `prompt_contract().format`, `prompt_contract` 외 1개
- 반환 표현: `prompt_contract().format(schema=schema, endpoints=quoted(endpoints), transcript=quoted(rows), pairs=quoted([[row['parent'], row['attempt']] for row in public if row['parent'] is not Non …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`classify` — L113**
```python
classify(text: str, finish_reason: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `credit._json_units`, `set`, `type`, `any`, `synthesis._single_line_scalar`, `synthesis._native_value_shape`, `note['corrected_facts'].items`, `_reference_shape`
- 반환 표현: `'TRUNCATED_PARTIAL_NOTEBOOK'`; `'INVALID_PARTIAL_NOTEBOOK'`; `'PARTIAL_NOTEBOOK_CAPTURED' if note['corrected_facts'] or note['lessons'] else 'NO_NOTEBOOK_CLAIMS'`

### D18.132 `fine_tuning/notebook_rl/resources.py`

[현재 소스](../../fine_tuning/notebook_rl/resources.py) · SHA256 `eb596771e70572c57699a999e96a7e30f4c8bfd854255c61e590e7e4f40e044e`

모듈의 기존 설명: Pinned resources available in either the original checkout or a folder copy.

**`resource_path` — L15**
```python
resource_path(descriptor: str, *, project_root: Path | None=None) -> Path
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `ValueError`, `Path`, `legacy.exists`, `legacy.is_symlink`, `_safe_root`, `_read`
- 반환 표현: `path`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.133 `fine_tuning/notebook_rl/rollout.py`

[현재 소스](../../fine_tuning/notebook_rl/rollout.py) · SHA256 `8aa08909c3c6c2a5832c4c52dc1a93f6a60d178ef12911f1d50ff149f5c8b16f`

모듈의 기존 설명: Same-state retry trees. Observed failures are data, not labels for SFT.

**클래스 `RolloutConfig` — L24**

기저: `object`.

```python
n: dict[int, int]
branch: dict[int, int]
max_attempts: int
max_new_tokens: dict[int, int]
context_limit: int
diag: bool
n_diag: int
oracle_counterfactual: bool = False
self_perception_probe: bool = False
diag_max_new_tokens: int = 256
visual_lambda: float = 0.0
master_seed: int = 3407
final_notebook: bool = False
```

**`RolloutConfig.from_protocol` — L40**
```python
RolloutConfig.from_protocol(cls, protocol: dict, algorithm: str, *, master_seed: int=3407) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `cls`, `r.get`
- 반환 표현: `cls(r['n'], r['branch'], 1 if algorithm == 'ALG-T1' else 3 if algorithm == 'ALG-NB3' else 2, r['max_new_tokens'], protocol['context_limit'], algorithm in r['diag']['enabled_in'], r['dia …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`RolloutConfig.validate` — L50**
```python
RolloutConfig.validate(self) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `any`, `self.n.get`, `range`, `self.max_new_tokens.get`, `self.branch.get`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`seeds` — L64**
```python
seeds(problem_id: str, policy_version: int, turn: int, branch: str, n: int, *, master_seed: int=3407) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `int.from_bytes`, `hashlib.sha256(f'{master_seed}|{problem_id}|{policy_version}|{turn}|{branch}|{k}'.encode() …`, `hashlib.sha256`, `f'{master_seed}|{problem_id}|{policy_version}|{turn}|{branch}|{k}'.encode`, `range`
- 반환 표현: `[int.from_bytes(hashlib.sha256(f'{master_seed}|{problem_id}|{policy_version}|{turn}|{branch}|{k}'.encode()).digest()[:8], 'big') % 2 ** 31 for k in range(n)]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_hash` — L73**
```python
_hash(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).e …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()`

**`select_failures` — L77**
```python
select_failures(samples: list[Sample], k: int, key: str) -> list[Sample]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `sorted`, `hashlib.sha256(f'{key}:{s.sample_id}'.encode()).digest`, `hashlib.sha256`, `f'{key}:{s.sample_id}'.encode`, `set`, `distinct.append`, `seen.add`, `repeated.append`
- 반환 표현: `(distinct + repeated)[:k]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`continuation_ids` — L94**
```python
continuation_ids(sample: Sample, feedback: str, tokenizer: Any) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tokenizer.convert_tokens_to_ids`, `type`, `getattr`, `ValueError`, `tokenizer.encode`
- 반환 표현: `sample.prompt_token_ids + sample.completion_token_ids + closure + suffix`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_sample_request` — L104**
```python
_sample_request(request: RolloutRequest, engine: Any, policy_version: int, context_limit: int, *, on_completion: Callable[[str, int, int, Completion], None] | None=None) -> RolloutResult
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `engine.prepare`, `len`, `ValueError`, `RuntimeError`, `engine.sample`
- 반환 표현: `outputs[0]`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`build_tree` — L123**
```python
build_tree(actor: ActorItem, images: list, engine: Any, verifier: Any, prompts: dict, cfg: RolloutConfig, policy_version: int, route_rule: str, *, on_group: Callable[[Group], None] | None=None, should_stop: Callable[[], bool] | None=None) -> ProblemTree
```
- 기존 설명: Solution tree first; DIAG is deferred to a class-balanced batch operation.
- 직접 호출: `cfg.validate`, `to_json`, `prompt_builders.for_actor`, `len`, `ValueError`, `generate`, `select_failures`, `range`, `prompt_builders.retry_text`, `continuation_ids` 외 10개
- 반환 표현: `tree`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`append_final_notebook` — L229**
```python
append_final_notebook(tree: ProblemTree, images: list, engine: Any, prompts: dict, cfg: RolloutConfig, *, should_stop: Callable[[], bool] | None=None) -> dict
```
- 기존 설명: Preserve one post-recovery model note, independently of balanced DIAG training.
- 직접 호출: `cfg.validate`, `to_json`, `ValueError`, `sorted`, `_hash`, `should_stop`, `continuation_ids`, `prompt_builders.diagnose_text`, `len`, `RolloutRequest` 외 4개
- 반환 표현: `record`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`append_diagnoses` — L274**
```python
append_diagnoses(trees: list[ProblemTree], images: dict[str, list], engine: Any, verifier: Any, prompts: dict, cfg: RolloutConfig, *, on_group: Callable[[Group], None] | None=None, should_stop: Callable[[], bool] | None=None) -> dict
```
- 기존 설명: Choose <=50% per verified class BEFORE generating; never drop after seeing rewards.
- 직접 호출: `cfg.validate`, `len`, `set`, `any`, `t.notebook.get`, `ValueError`, `to_json`, `tree.notebook.get`, `from_json`, `json.dumps` 외 27개
- 반환 표현: `report`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.134 `fine_tuning/notebook_rl/schema.py`

[현재 소스](../../fine_tuning/notebook_rl/schema.py) · SHA256 `4c12aea4d0c23792afdd37e57ad62697d3cbc7766bd2700079d45512fea08c93`

모듈의 기존 설명: Strict wire records for the v2 notebook-RL design; no model dependencies.

**클래스 `ActorItem` — L26**

기저: `object`.

```python
problem_id: str
source: str
family: str
mode: Mode
question: str
choices: tuple[str, ...]
answer_format: AnswerType
image_paths: tuple[str, ...]
image_sha256: tuple[str, ...]
```

**클래스 `VerifierItem` — L39**

기저: `object`.

```python
problem_id: str
answer_type: AnswerType
gold: str
accept: dict
choices: tuple[str, ...]
slots: dict[str, str]
slot_types: dict[str, str]
required_slots: tuple[str, ...]
oracle_fact_lines: tuple[str, ...]
printed_text: tuple[str, ...]
```

**클래스 `Lineage` — L53**

기저: `object`.

```python
problem_id: str
source: str
source_revision: str
license: str
label_provenance: Literal['PROGRAM', 'HUMAN']
image_provenance: str
upstream_origin: str
generator_version: str | None
generator_seed: int | None
template_id: str
structure_hash: str
split: Split
parents: tuple[str, ...]
benchmark_derived: bool = False
model_touched: bool = False
```

**클래스 `SlotReport` — L72**

기저: `object`.

```python
parsed: int
required: int
required_correct: int
wrong_required: tuple[str, ...]
all_correct: int
all_total: int
conflicting: tuple[str, ...]
false_relations: tuple[str, ...] = ()
unverified_relations: tuple[str, ...] = ()
```

**클래스 `VerifierResult` — L85**

기저: `object`.

```python
status: Status
extracted: str | None
reward_answer: float
slots: SlotReport | None
verifier_version: str
```

**클래스 `Sample` — L94**

기저: `object`.

```python
sample_id: str
problem_id: str
group_id: str
kind: GroupKind
turn: int
parent_sample_id: str | None
prompt_token_ids: list[int]
completion_token_ids: list[int]
sampler_logprobs: list[float]
finish_reason: str
text: str
seed: int
policy_version: int
result: VerifierResult | None = None
diag_reward: float | None = None
reward: float = 0.0
```

**클래스 `Group` — L114**

기저: `object`.

```python
group_id: str
kind: GroupKind
problem_id: str
state_sha256: str
samples: list[Sample]
usable: bool
reason: str | None
```

**클래스 `ProblemTree` — L125**

기저: `object`.

```python
problem_id: str
policy_version: int
groups: list[Group]
route: Route
causes: dict[str, Cause]
notebook: dict
```

**클래스 `DiagTarget` — L135**

기저: `object`.

```python
cause3: Literal['VISUAL', 'NON_VISUAL', 'FINALIZATION']
wrong_slots: tuple[str, ...]
```

**클래스 `OracleProbe` — L141**

기저: `object`.

```python
probe_id: str
problem_id: str
kind: Literal['oracle', 'self_perception']
prompt_token_ids: list[int]
completion_token_ids: list[int]
status: Status
policy_version: int
```

**`_decode` — L157**
```python
_decode(hint: Any, value: Any, at: str) -> Any
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `get_origin`, `get_args`, `json.dumps`, `_decode`, `ValueError`, `any`, `type`, `fields`, `value.keys`, `sorted` 외 9개
- 반환 표현: `value`; `_decode(candidate, value, at)`; `obj`; 외 3개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_actor` — L206**
```python
validate_actor(a: ActorItem) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `a.question.strip`, `all`, `x.strip`, `len`, `zip`, `PurePosixPath`, `p.is_absolute`, `PureWindowsPath` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_record` — L229**
```python
_validate_record(obj: Any) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `validate_actor`, `obj.gold.strip`, `ValueError`, `set`, `obj.slots.keys`, `len`, `obj.slot_types.values`, `any`, `slot_id` 외 1개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_unique_pairs` — L301**
```python
_unique_pairs(pairs: list[tuple[str, Any]]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_invalid_constant` — L310**
```python
_invalid_constant(value: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`from_json` — L314**
```python
from_json(cls: type[T], s: str) -> T
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `json.loads`, `_decode`
- 반환 표현: `_decode(cls, value, cls.__name__)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`to_json` — L324**
```python
to_json(obj: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `json.dumps`, `asdict`, `from_json`
- 반환 표현: `json.dumps(asdict(normalized), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_protocol` — L334**
```python
load_protocol(path: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.read_bytes`, `yaml.safe_load`, `isinstance`, `data.get`, `ValueError`, `data.get('initial_checkpoint', {}).get`, `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`
- 반환 표현: `data`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.135 `fine_tuning/notebook_rl/shared_policy_engine.py`

[현재 소스](../../fine_tuning/notebook_rl/shared_policy_engine.py) · SHA256 `007463014800d3817f325208e990b709f80117fe24231d43180e3be0eb5526e2`

모듈의 기존 설명: Sequential HF smoke rollouts using the trainer's live, unmerged PEFT policy.

**클래스 `SharedPolicyRolloutEngine` — L13**

기저: `HFRolloutEngine`.

**`SharedPolicyRolloutEngine.__init__` — L20**
```python
SharedPolicyRolloutEngine.__init__(self, bundle: PolicyBundle, image_cfg: dict, max_model_len: int, *, logprob_chunk: int=1024, policy_version: int=0) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `hasattr`, `type`, `ValueError`, `InputPreparation.__init__`, `id`, `tuple`, `self.model.named_parameters`, `bundle.params.items`, `self._assert_owner`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`SharedPolicyRolloutEngine._assert_owner` — L37**
```python
SharedPolicyRolloutEngine._assert_owner(self, bundle: PolicyBundle) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `id`, `tuple`, `self.model.named_parameters`, `bundle.params.items`, `ValueError`, `self._partitions.values`, `set`, `len`, `self._parameters.values`, `any` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`SharedPolicyRolloutEngine.mark_updated` — L55**
```python
SharedPolicyRolloutEngine.mark_updated(self, bundle: PolicyBundle, version: int) -> None
```
- 기존 설명: Invalidate certification; shared storage alone never certifies sampling.
- 직접 호출: `self._assert_owner`, `type`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`SharedPolicyRolloutEngine.initialize_weights` — L63**
```python
SharedPolicyRolloutEngine.initialize_weights(self, named_weights: Iterable[tuple[str, torch.Tensor]], policy_version: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `RuntimeError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`SharedPolicyRolloutEngine.push_weights` — L66**
```python
SharedPolicyRolloutEngine.push_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `RuntimeError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`SharedPolicyRolloutEngine._load_weights` — L69**
```python
SharedPolicyRolloutEngine._load_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `RuntimeError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`SharedPolicyRolloutEngine.sample_with_settings` — L72**
```python
SharedPolicyRolloutEngine.sample_with_settings(self, requests: list[RolloutRequest], sampling: SamplingSettings, *, on_completion: Callable[[str, int, int, Completion], None] | None=None) -> list[RolloutResult]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._assert_owner`, `next`, `self.model.parameters`, `torch.autocast`, `nullcontext`, `super().sample_with_settings`, `super`, `self.bundle.route.apply`
- 반환 표현: `super().sample_with_settings(requests, sampling)`; `super().sample_with_settings(requests, sampling, on_completion=on_completion)`

**`SharedPolicyRolloutEngine.probe_logprobs` — L86**
```python
SharedPolicyRolloutEngine.probe_logprobs(self, request: RolloutRequest, completion_ids: list[int]) -> list[float]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._assert_owner`, `ValueError`, `next`, `self.model.parameters`, `self.model_inputs`, `self.model.eval`, `torch.no_grad`, `completion_logprobs(self.bundle, batch, len(completion_ids), chunk=self.logprob_chunk)[0]. …`, `completion_logprobs(self.bundle, batch, len(completion_ids), chunk=self.logprob_chunk)[0]. …`, `completion_logprobs` 외 3개
- 반환 표현: `completion_logprobs(self.bundle, batch, len(completion_ids), chunk=self.logprob_chunk)[0].cpu().tolist()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.136 `fine_tuning/notebook_rl/shared_smoke.py`

[현재 소스](../../fine_tuning/notebook_rl/shared_smoke.py) · SHA256 `79e6c24a51905c1cbfef2dff69a0ee39d256ecaf5d7ce16d45db7c54dccf45e7`

모듈의 기존 설명: Bounded shared-policy HF smoke for 24GB GPUs, never general training approval.

**`memory_floor` — L32**
```python
memory_floor(protocol: dict, weight_bytes: int) -> int
```
- 기존 설명: Conservative persistent allowance, NOT an activation/peak-fit certificate.
- 직접 호출: `type`, `ValueError`, `set`, `any`, `counts.values`, `sum`
- 반환 표현: `weight_bytes + 16 * sum(counts.values())`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_shape` — L43**
```python
_shape(job: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `job['runtime_scope'].get`, `len`, `any`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_job` — L54**
```python
validate_job(job: dict, protocol: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_shape`, `syntax.update`, `gate_session.validate_job`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`preflight` — L63**
```python
preflight(job: dict, protocol: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_job`, `approval.validate_for_gate_data`, `Path`, `gate_session._eligible`, `gate_session._probe_actors`, `gate_session._actors`, `set`, `actors.keys`, `probes.values`, `len` 외 1개
- 반환 표현: `checked`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_binding` — L78**
```python
_binding(job: dict, protocol: dict, checked: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `launch.checkpoint_binding`, `digest`
- 반환 표현: `result`

**`_final_notes` — L87**
```python
_final_notes(batch, actors, image_root, engine, prompts, cfg, should_stop) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `trees.append`, `from_json`, `load_images`, `append_final_notebook`, `image.close`, `to_json`
- 반환 표현: `{t.problem_id: t.notebook['final_notebook']['status'] for t in trees}`

**`_prepare_hf_dependencies` — L113**
```python
_prepare_hf_dependencies() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `is_bnb_available`, `import_module`
- 반환 표현: `{value.__name__: value.__module__ for value in (LoraConfig, get_peft_model, AutoProcessor, Qwen3VLForConditionalGeneration)}`

**`execute` — L128**
```python
execute(job: dict, protocol: dict, checked: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `gate_session._eligible`, `preflight`, `ValueError`, `launch.runtime_check`, `_prepare_hf_dependencies`, `Path`, `_binding`, `_safe_root`, `output.mkdir`, `_write` 외 52개
- 반환 표현: `result`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`reload_check` — L261**
```python
reload_check(job: dict, protocol: dict) -> dict
```
- 기존 설명: Run as a separate CLI process AFTER capture released its only model copy.
- 직접 호출: `_shape`, `_safe_root`, `Path`, `launch.read_json`, `closed.get`, `type`, `os.getpid`, `psutil.pid_exists`, `ValueError`, `_binding` 외 13개
- 반환 표현: `verified`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L303**
```python
main(argv=None) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.add_mutually_exclusive_group`, `mode.add_argument`, `parser.parse_args`, `launch.read_json`, `load_protocol`, `print`, `json.dumps`, `reload_check` 외 5개
- 반환 표현: `0`; `0 if gate_session._eligible(checked) else 2`; `0 if result['status'] == 'CAPTURED_PENDING_FRESH_RELOAD' else 2`; 외 1개 분기

### D18.137 `fine_tuning/notebook_rl/smoke_data_evidence.py`

[현재 소스](../../fine_tuning/notebook_rl/smoke_data_evidence.py) · SHA256 `96158c33ca7fdc2bcb320cb080eb268183c1e69a1bb158db97b0840191ee7f03`

모듈의 기존 설명: Actual CPU data measurements for the separately authorized, program-only smoke.

**`_sha` — L37**
```python
_sha(raw) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`
- 반환 표현: `hashlib.sha256(raw).hexdigest()`

**`_hash` — L41**
```python
_hash(value) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_bytes`
- 반환 표현: `_sha(_bytes(value))`

**`_json` — L45**
```python
_json(path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `_read`, `Path`
- 반환 표현: `json.loads(_read(Path(path)), object_pairs_hook=_unique)`

**`_descriptor` — L49**
```python
_descriptor(path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`, `Path(path).absolute`, `Path`, `_sha`, `_read`
- 반환 표현: `{'path': str(Path(path).absolute()), 'file_sha256': _sha(_read(Path(path)))}`

**`_bound_json` — L53**
```python
_bound_json(descriptor) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_read`, `Path`, `_sha`, `ValueError`, `json.loads`
- 반환 표현: `json.loads(raw, object_pairs_hook=_unique)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_inputs` — L60**
```python
_inputs(root) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_safe_root`, `_json`, `read_actors`, `read_lineage`, `read_verifier`, `set`, `ValueError`, `str`, `_sha`, `_read` 외 3개
- 반환 표현: `(items, files, manifests)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_revision` — L88**
```python
_revision(module) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_read`, `Path`, `b''.join`, `getattr`
- 반환 표현: `_sha(_read(Path(module.__file__)) + b'\x00' + _read(_HERE / 'generators/common.py') + b''.join((b'\x00' + _read(Path(dep.__file__)) for dep in getattr(module, 'SOURCE_DEPENDENCIES', ()) …`

**`_solvers` — L93**
```python
_solvers(world, template) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `template.rsplit`, `graph._validate_world`, `graph._answers`
- 반환 표현: `graph._answers(world, traversal)`

**`intervene` — L99**
```python
intervene(world, template) -> unannotated
```
- 기존 설명: One predetermined, label-changing structural/weight intervention; no model selection.
- 직접 호출: `copy.deepcopy`, `_solvers`, `set`, `accepting.symmetric_difference_update`, `sorted`, `accepting.add`, `len`, `accepting.remove`, `graph._bfs`, `next` 외 4개
- 반환 표현: `changed`

**`measure_interventions` — L143**
```python
measure_interventions(source, template, example, output, *, count=100) -> unannotated
```
- 기존 설명: Keep every attempted pair, including errors and unchanged answers, on disk.
- 직접 호출: `get_generator`, `int`, `actor.problem_id.split`, `range`, `module.generate`, `module.self_check`, `ValueError`, `intervene`, `_solvers`, `graph_pair._render` 외 20개
- 반환 표현: `{'pairs': len(rows), 'answer_changed': sum((row['status'] == 'PASS' for row in rows)), 'question_unchanged': all((row.get('question_unchanged') is True for row in rows)), 'nonimage_inpu …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_guard_worker` — L198**
```python
_guard_worker(root, benchmark) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `install_guards`, `private.read_bytes`, `benchmark.read_bytes`, `__import__`, `action`, `os.getpid`
- 반환 표현: `{'pid': os.getpid(), 'measurements': result}`

**`measure_isolation` — L215**
```python
measure_isolation(root, benchmark) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `benchmark.exists`, `ValueError`, `subprocess.run`, `str`, `json.loads`, `os.getpid`, `all`, `record['measurements'].values`
- 반환 표현: `record`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`produce_components` — L228**
```python
produce_components(root, review_descriptor, benchmark, output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_safe_root`, `Path`, `artifact_path`, `output.exists`, `FileExistsError`, `_inputs`, `str`, `_sha`, `_read`, `_json` 외 39개
- 반환 표현: `result`
- 명시적 예외: `FileExistsError`, `ValueError`; 전파되는 callee 예외는 별도.

**`assemble` — L306**
```python
assemble(components_descriptor, root, receipt_descriptor, binding, scope, output, *, source_registry, metadata_rebind=None) -> unannotated
```
- 기존 설명: Bind measured components only after real exclusion; never create GPU approval.
- 직접 호출: `_bound_json`, `_safe_root`, `Path`, `_inputs`, `_json`, `_descriptor`, `_hash`, `ValueError`, `any`, `_sha` 외 34개
- 반환 표현: `report`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L393**
```python
main(argv=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `produce_components`, `str`, `print`, `json.dumps`
- 반환 표현: `0`

### D18.138 `fine_tuning/notebook_rl/synthesis_credit.py`

[현재 소스](../../fine_tuning/notebook_rl/synthesis_credit.py) · SHA256 `dbc038f5194450466856a9dcf7b14f7cee2dedb6d37032b87b095dcc92de8487`

모듈의 기존 설명: Host-only credit for independently verified submitted facts and relationships.

**`contract` — L43**
```python
contract(version: str=VERSION) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `contract`
- 반환 표현: `{**contract(), 'objective_version': version, 'assessment_weight': ASSESSMENT_WEIGHT, 'formula': contract()['formula'] + ' + 0.0625*correct_native_assessments/all_public_native_refs', 'a …`; `{'objective_version': VERSION, 'strict_reward_version': synthesis.BOUNDED_VERSION, 'strict_weight': 1.0, 'fact_weight': FACT_WEIGHT, 'relationship_weight': RELATIONSHIP_WEIGHT, 'formula …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_bad_constant` — L82**
```python
_bad_constant(value: str) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_json_units` — L86**
```python
_json_units(text: str) -> tuple[dict | None, list[tuple], list[tuple]]
```
- 기존 설명: Walk decoder offsets, never search for a repeated textual value or object.
- 직접 호출: `json.JSONDecoder`, `decoder.decode`, `type`, `members`, `whitespace`, `note.get`, `list`, `elements`
- 반환 표현: `(None, [], [])`; `(note, facts, lessons)`

**`issue_id` — L129**
```python
issue_id(attempt: int, row: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps`
- 반환 표현: `json.dumps([attempt, row['kind'], row['rule'], row['locus']], separators=(',', ':'))`

**`_origin` — L133**
```python
_origin(lesson: Any) -> str | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `lesson.get`, `earlier.get`, `any`, `issue_id`
- 반환 표현: `None`; `issue_id(earlier['attempt'], earlier)`

**`_selected` — L141**
```python
_selected(lessons: list[tuple], original_ids: list[str], public_index: dict | None=None, paired_index: dict | None=None) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `enumerate`, `_origin`, `type`, `lesson.get`, `origins.append`, `public_index.get`, `paired_index.get`
- 반환 표현: `selected`

**`_credit_lesson` — L156**
```python
_credit_lesson(value: Any, public_index: dict | None, paired_index: dict | None=None) -> Any
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `paired_expand`, `reference_expand`
- 반환 표현: `paired_expand(value, paired_index)`; `value`; `reference_expand(value, public_index)`

**`_paired_lessons` — L168**
```python
_paired_lessons(note: dict | None, lessons: list[tuple]) -> list[tuple]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `type`
- 반환 표현: `[]`; `lessons`

**`_pair_action_units` — L175**
```python
_pair_action_units(text: str, finish_reason: str, target: dict, originals: list[str]) -> tuple[dict, dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_origin`, `parse_label`, `len`, `text.lstrip`, `text.rstrip`
- 반환 표현: `({}, {}, {})`; `({origin: 0}, spans, {origin: label})`

**`_pair_action_metadata` — L186**
```python
_pair_action_metadata(tree, sample_id: str, target: dict, solution: str, transcript: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `digest`, `action.contract`, `ValueError`, `copy.deepcopy`
- 반환 표현: `{'notebook_generation_policy': action.VERSION, 'pair_action_binding': binding, 'pair_action_binding_sha256': digest({'binding': binding, 'solution_sha256': solution, 'transcript_sha256' …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_pair_action_binding` — L204**
```python
_validate_pair_action_binding(row: dict, sample) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `type`, `set`, `any`, `action.contract`, `digest`, `ValueError`, `_hash`, `len`, `enumerate` 외 5개
- 반환 표현: `target`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_reference_metadata` — L257**
```python
_reference_metadata(public_index: dict, solution_sha256: str, transcript_sha256: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `digest`
- 반환 표현: `metadata | {'public_endpoints_binding_sha256': digest({'notebook_generation_policy': reference_version, 'public_endpoints_sha256': metadata['public_endpoints_sha256'], 'solution_sha256' …`

**`_paired_metadata` — L267**
```python
_paired_metadata(public_index: dict, solution_sha256: str, transcript_sha256: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `digest`
- 반환 표현: `metadata | {'public_pairs_binding_sha256': digest({'notebook_generation_policy': paired_version, 'public_pairs_sha256': metadata['public_pairs_sha256'], 'solution_sha256': solution_sha2 …`

**`_validate_paired_index` — L277**
```python
_validate_paired_index(row: dict) -> dict
```
- 기존 설명: Validate receipt bookkeeping; fresh scoring must still rebuild all public pairs.
- 직접 호출: `row.get`, `type`, `any`, `ValueError`, `public_index.items`, `set`, `pair.values`, `_range_shape`, `len`, `tuple` 외 3개
- 반환 표현: `public_index`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_reference_index` — L316**
```python
_validate_reference_index(row: dict) -> dict | None
```
- 기존 설명: Check saved public metadata; optimizer callers still regenerate it from actual solutions.
- 직접 호출: `any`, `row.get`, `type`, `ValueError`, `public_index.items`, `re.fullmatch`, `set`, `int`, `_range_shape`, `len` 외 9개
- 반환 표현: `None`; `public_index`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_range_shape` — L355**
```python
_range_shape(value: Any) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `all`, `len`
- 반환 표현: `type(value) is list and all((type(pair) is list and len(pair) == 2 and all((type(number) is int for number in pair)) for pair in value))`

**`_lesson_shape` — L360**
```python
_lesson_shape(value: Any) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `_range_shape`, `any`, `row.values`
- 반환 표현: `False`; `True`

**`_fact_candidates` — L389**
```python
_fact_candidates(note: dict | None, facts: list[tuple]) -> dict[str, tuple[str, list[int]]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `note.get`, `slot_id`, `len`, `set`, `synthesis._single_line_scalar`, `synthesis._native_value_shape`
- 반환 표현: `{}`; `{slot_id(key): (value, [start, end]) for key, value, start, end in facts if synthesis._single_line_scalar(key) and synthesis._single_line_scalar(value) and synthesis._native_value_shape …`

**`_seal` — L400**
```python
_seal(row: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `digest`
- 반환 표현: `row | {'score_sha256': digest(row)}`

**`_strict_digest` — L404**
```python
_strict_digest(value: dict) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `digest`, `typed`
- 반환 표현: `digest(value)`; `digest(typed(value))`

**`_whole_reference_metadata` — L418**
```python
_whole_reference_metadata(samples, expanded: dict, solution: str, transcript: str, policy_version: int) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `reference.contract`, `getattr`, `_hash`, `digest`, `hashlib.sha256(rendered.encode()).hexdigest`, `hashlib.sha256`, `rendered.encode`
- 반환 표현: `{'notebook_generation_policy': reference.VERSION, 'whole_reference_binding': binding, 'whole_reference_binding_sha256': digest({'binding': binding, 'solution_sha256': solution, 'transcr …`

**`_whole_reference_units` — L433**
```python
_whole_reference_units(text: str, expanded: dict) -> tuple[dict | None, list[tuple], list[tuple]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_json_units`, `expanded['decisions'].items`
- 반환 표현: `(None, [], [])`; `(note, facts, [(lesson, *expanded['decision_spans'][ref]) for ref, lesson in expanded['decisions'].items()])`

**`_assessment_states` — L443**
```python
_assessment_states(inventory: dict, audit: dict) -> dict
```
- 기존 설명: Expected claims are host-only; an absent claim never certifies compliance.
- 직접 호출: `references.items`, `synthesis._key`, `synthesis._eligible_unknown`, `len`, `ValueError`, `index.get`, `native.index`, `dependencies.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_assessment_units` — L470**
```python
_assessment_units(text: str, finish_reason: str, inventory: dict) -> tuple[bool, dict, dict, dict]
```
- 기존 설명: Decode original literals independently of header and other row validity.
- 직접 호출: `reference._decode`, `type`, `set`, `len`, `enumerate`, `zip`, `all`, `copy.deepcopy`
- 반환 표현: `(False, {}, {}, {})`; `(True, rows, states, spans)`

**`_assessment_state_matches` — L498**
```python
_assessment_state_matches(ref: str, rows: dict, expected: dict, inventory: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `matches`, `inventory['references'].get`, `inventory['assessment_refs'].index`, `source['rule'].startswith`, `target['rule'].startswith`
- 반환 표현: `False`; `True`

**`_assessment_decision` — L529**
```python
_assessment_decision(ref: str, rows: dict, inventory: dict, expected: dict) -> tuple[bool, dict | None]
```
- 기존 설명: Expand one authored decision using existing public grammar, without rewriting it.
- 직접 호출: `_assessment_state_matches`, `type`, `len`, `reference._endpoint`, `str`
- 반환 표현: `(False, None)`; `(True, None)`; `(True, {'rule': source['rule'], 'earlier': reference._endpoint(source), 'later': target, 'relation': relation})`

**`_score_assessments` — L571**
```python
_score_assessments(score: dict, text: str, finish_reason: str, expanded: dict, samples, fresh: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_assessment_states`, `_assessment_units`, `dict.fromkeys`, `_assessment_decision`, `int`, `synthesis._lesson`, `issue_id`, `inventory['references'].items`, `score['relationship_rewards'].items`, `score['relationship_later_answer_correct'].pop` 외 9개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_whole_reference_sample_binding` — L599**
```python
_whole_reference_sample_binding(sample) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `digest`
- 반환 표현: `{'sample_id': sample.sample_id, 'group_id': sample.group_id, 'policy_version': sample.policy_version, 'prompt_token_ids_sha256': digest(sample.prompt_token_ids), 'completion_token_ids_s …`

**`_validate_whole_reference_binding` — L606**
```python
_validate_whole_reference_binding(row: dict, sample) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `type`, `set`, `reference.contract`, `_whole_reference_sample_binding`, `any`, `ValueError`, `SimpleNamespace`, `reference.expand`, `_whole_reference_metadata` 외 1개
- 반환 표현: `expanded`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_score_submission` — L633**
```python
_score_submission(text, finish_reason, tree, fresh, strict, verifier, actor, note, facts, lessons, public_index=None, paired_index=None, action_target=None, *, original=None) -> dict
```
- 기존 설명: Shared existing unit scoring on submitted grading text; never creates an action.
- 직접 호출: `episode.audit_episode`, `verifier.grade`, `type`, `ValueError`, `issue_id`, `len`, `set`, `episode.solution_samples`, `_fact_candidates`, `candidates.items` 외 21개
- 반환 표현: `score`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`score_note` — L713**
```python
score_note(text, finish_reason, tree, audit, verifier, strict_report=None, *, actor, objective_version: str=VERSION, replay_relation_v1: bool=False, sample_id: str | None=None, observed_sample=None) -> dict
```
- 기존 설명: Fresh strict and unit verification; observed policy1 samples remain diagnostic.
- 직접 호출: `ValueError`, `tree.notebook.get('rewarded_notebook', {}).get`, `tree.notebook.get`, `synthesis.audit_episode`, `type`, `_hash`, `strict_grade`, `_strict_digest`, `episode.solution_samples`, `any` 외 19개
- 반환 표현: `_seal(score)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_score_pair_notes` — L798**
```python
_score_pair_notes(tree, audit, verifier, strict_reports, *, actor) -> dict[str, dict]
```
- 기존 설명: Reuse fresh evidence only within one already binding-checked pair-label batch.
- 직접 호출: `synthesis.audit_episode`, `type`, `_hash`, `ValueError`, `episode.audit_episode`, `action._grade_note_audited`, `_strict_digest`, `action._bound_target_checked`, `_json_units`, `_score_submission` 외 3개
- 반환 표현: `scored`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_raw_page` — L826**
```python
_raw_page(sample) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads(to_json(sample)).items`, `json.loads`, `to_json`
- 반환 표현: `{key: value for key, value in json.loads(to_json(sample)).items() if key not in ('result', 'diag_reward', 'reward')}`

**`_page_layout` — L832**
```python
_page_layout(tree, observed_groups=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tree.notebook.get`, `episode.solution_samples`, `pages.plan`, `note.get`, `type`, `len`, `ValueError`, `any`, `all_ids.update`, `enumerate` 외 6개
- 반환 표현: `(originals, expected_plan, calls, selected)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_project_page_score` — L885**
```python
_project_page_score(common, sample, assembled, page_index, binding) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `facts.items`, `next`, `decisions.values`, `common['relationship_spans'].items`, `any`, `common['fact_rewards'].items`, `common['relationship_rewards'].items`, `sum`, `fact_rewards.values`, `relationships.values` 외 11개
- 반환 표현: `_seal(row)`

**`score_page_bundles` — L921**
```python
score_page_bundles(tree, audit, verifier, *, actor, observed_groups=None) -> dict[str, dict]
```
- 기존 설명: Regrade four complete ordinal bundles; credit only their actual owned page literals.
- 직접 호출: `episode.validate_update_binding`, `_page_layout`, `synthesis.audit_episode`, `type`, `_hash`, `ValueError`, `range`, `pages.assemble`, `reference.grade_note`, `_whole_reference_units` 외 9개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_page_score` — L956**
```python
_validate_page_score(row, sample) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `type`, `set`, `pages.contract`, `digest`, `any`, `ValueError`, `SimpleNamespace`, `Sample`, `pages.assemble` 외 12개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_submission_fields` — L1018**
```python
_validate_submission_fields(row, text, finish_reason, note, fact_units, lessons, originals, credit_types, public_index=None, paired_index=None, action_target=None) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_fact_candidates`, `_selected`, `selected.items`, `_lesson_shape`, `_credit_lesson`, `type`, `lessons[index][0].get`, `_pair_action_units`, `row.get`, `any` 외 10개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_validate_assessments` — L1053**
```python
_validate_assessments(row: dict, sample, expanded: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `_assessment_units`, `type`, `len`, `set`, `any`, `rewards.values`, `row['assessment_spans'].values`, `ValueError`, `issue_id` 외 7개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_score` — L1099**
```python
validate_score(row: dict, sample) -> None
```
- 기존 설명: Check receipt math, raw spans and sample binding; callers must also freshly regrade semantics.
- 직접 호출: `type`, `row.get`, `any`, `digest`, `contract`, `row.items`, `ValueError`, `re.fullmatch`, `hashlib.sha256(sample.text.encode()).hexdigest`, `hashlib.sha256` 외 20개
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_group` — L1189**
```python
_group(samples, scores: dict[str, dict]) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `any`, `ValueError`, `zip`, `validate_score`, `row.get`, `rows[0].get`, `row['page_binding'].items`, `list`, `range` 외 1개
- 반환 표현: `rows`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`has_signal` — L1222**
```python
has_signal(samples, scores: dict[str, dict]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_group`, `len`, `any`, `row['fact_rewards'].get`
- 반환 표현: `len({row['strict_reward'] for row in rows}) > 1 or any((len({row['fact_rewards'].get(key, 0) for row in rows}) > 1 for key in facts)) or any((len({row['relationship_rewards'][key] for r …`

**`group_credit` — L1232**
```python
group_credit(samples, scores: dict[str, dict], tokenizer) -> dict[str, dict]
```
- 기존 설명: Center all original samples before MIS; map only original, closed submitted units.
- 직접 호출: `_group`, `sorted`, `sum`, `row['fact_rewards'].get`, `digest`, `zip`, `token_spans`, `len`, `range`, `enumerate` 외 4개
- 반환 표현: `output`

### D18.139 `fine_tuning/notebook_rl/trainer.py`

[현재 소스](../../fine_tuning/notebook_rl/trainer.py) · SHA256 `ffce66cf190afae74a64321898b31d9dc6d7337999bc1fa6ae1ba459b34bdf7e`

모듈의 기존 설명: Bounded RL execution core. Call only after separate data/environment approval.

**`_write` — L30**
```python
_write(path: Path, value: dict) -> None
```
- 기존 설명: Write-once evidence, never overwrite an earlier failed/partial attempt.
- 직접 호출: `(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode`, `json.dumps`, `path.open`, `stream.write`, `stream.flush`, `os.fsync`, `stream.fileno`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**클래스 `SyncProbe` — L40**

기저: `object`.

```python
request: RolloutRequest
completion_ids: list[int]
```

**`synchronize` — L45**
```python
synchronize(bundle: PolicyBundle, preparation: Any, engine: Any, probes: list[SyncProbe], protocol: dict, step: int, *, push: bool) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `sorted`, `ValueError`, `isinstance`, `engine.mark_updated`, `engine.push_weights`, `merged_named_tensors`, `RuntimeError`, `next`, `bundle.model.parameters` 외 7개
- 반환 표현: `reports`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`grounding_disabled` — L72**
```python
grounding_disabled(history: list[dict], f0: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `r.get`, `f0.get`, `isinstance`, `math.isfinite`, `any`, `type`, `ValueError`, `len`, `all`
- 반환 표현: `False`; `len(values) >= 2 and all((v < baseline for v in values[-2:]))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`batch_statistics` — L84**
```python
batch_statistics(batch: CollectedBatch) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `from_json`, `len`, `statistics.median`, `sum`
- 반환 표현: `{'n_observed_solutions': len(solutions), 'len_median': statistics.median((len(s.completion_token_ids) for s in solutions)) if solutions else None, 'truncation_rate': sum((s.finish_reaso …`

**`_record` — L95**
```python
_record(batch: CollectedBatch, update: dict, algorithm: str, seed: int, step: int, push_seconds: float, dev: dict | None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `effective_rewards`, `metrics.step_record`, `len`, `statistics.mean`, `by_kind.items`, `sum`, `set`, `update['routes'].values`, `batch_statistics`, `dev.get`
- 반환 표현: `metrics.step_record(step=step, policy_version=step, n_problems=len(batch.trees), n_discarded=len(batch.discarded), n_groups_by_kind=update['n_groups_by_kind'], reward_mean_by_kind={k: s …`

**`run_loop` — L114**
```python
run_loop(*, bundle: PolicyBundle, optimizer: Any, scheduler: Any, engine: Any, preparation: Any, verifier: Any, actors: list[ActorItem], structures: dict[str, str], image_loader: Callable[[ActorItem], list], prompts: dict, protocol: dict, algorithm: str, seed: int, output: Path, binding: dict, f0_reference: dict, probes: list[SyncProbe], deadline_timestamp: float, max_steps: int, cursor: dict | None=None, dev_evaluator: Callable[[int], dict] | None=None, clock: Callable[[], float]=time.time) -> dict
```
- 기존 설명: Train one frozen epoch; dependency-injected core, not an approval bypass CLI.
- 직접 호출: `type`, `math.isfinite`, `ValueError`, `family_order`, `set`, `checkpoint.digest`, `to_json`, `hashlib.sha256(im.tobytes()).hexdigest`, `hashlib.sha256`, `im.tobytes` 외 42개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

### D18.140 `fine_tuning/notebook_rl/updates.py`

[현재 소스](../../fine_tuning/notebook_rl/updates.py) · SHA256 `64509a3a404167ed4f5b35d8f593a3d3c87e449b8509145d9d7186c5e795f8f7`

모듈의 기존 설명: One on-policy optimizer update, not a deployment or data-approval entry point.

**`_objective_identity` — L35**
```python
_objective_identity(tree: ProblemTree, objective: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `objective.get`, `isinstance`, `contract`, `ValueError`, `validate_update_binding`, `tree.notebook.get`, `note.get`
- 반환 표현: `identity`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_score_notebook_objective` — L66**
```python
_score_notebook_objective(tree: ProblemTree, actor: ActorItem, verifier: Any, objective: dict, *, replay_relation_v1: bool=False) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_objective_identity`, `note.get`, `ValueError`, `any`, `credit_objective.score_page_bundles`, `_hash`, `audit_episode`, `note['episode_audit'].get`, `tree.notebook.get`, `len` 외 8개
- 반환 표현: `receipt | {'sha256': _hash(receipt)}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`bind_notebook_objective` — L137**
```python
bind_notebook_objective(tree: ProblemTree, actor: ActorItem, verifier: Any, objective: dict) -> dict
```
- 기존 설명: Attach host-only rewards once; original sampled outcomes remain unchanged.
- 직접 호출: `ValueError`, `_score_notebook_objective`
- 반환 표현: `receipt`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`notebook_objective_scores` — L146**
```python
notebook_objective_scores(tree: ProblemTree, objective: dict | None) -> dict[str, dict]
```
- 기존 설명: Check recorded scores for selection; optimizer must also regrade them fresh.
- 직접 호출: `tree.notebook.get`, `ValueError`, `_objective_identity`, `type`, `set`, `any`, `identity.items`, `_hash`, `receipt.items`, `receipt['samples'].items` 외 4개
- 반환 표현: `{}`; `receipt['samples']`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`has_learning_signal` — L198**
```python
has_learning_signal(tree: ProblemTree, *, include_notebooks: bool=False, notebook_objective: dict | None=None) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `tree.notebook.get`, `note.get`, `validate_update_binding`, `note['episode_audit'].get`, `notebook_objective_scores`, `any`, `group_has_signal`
- 반환 표현: `False`; `any((g.usable and (g.kind != 'DIAG' or g.group_id in notebook_groups) and group_has_signal(g, scores) for g in tree.groups))`

**`group_has_signal` — L220**
```python
group_has_signal(group: Group, scores: dict[str, dict]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `all`, `has_signal`, `sorted`, `len`, `row['fact_rewards'].get`, `zip`
- 반환 표현: `has_signal(group.samples, scores)`; `len({(s.reward, *(row['fact_rewards'].get(k, 0) for k in keys)) for s, row in zip(group.samples, rows, strict=True)}) > 1`; `len({row['effective_reward'] for row in rows}) > 1`; 외 1개 분기

**`notebook_signal_groups` — L238**
```python
notebook_signal_groups(tree: ProblemTree, scores: dict[str, dict]) -> int
```
- 기존 설명: Count one complete paged notebook set as one logical DIAG signal group.
- 직접 호출: `sum`, `group_has_signal`, `tree.notebook.get('rewarded_notebook', {}).get`, `tree.notebook.get`, `int`, `bool`
- 반환 표현: `int(bool(count)) if tree.notebook.get('rewarded_notebook', {}).get('notebook_generation_policy') == pages_version else count`

**`_page_memberships` — L246**
```python
_page_memberships(receipts: dict) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `receipts.items`, `receipt.get`, `type`, `ValueError`, `rows.items`, `row.get`, `Sample`, `synthesis_credit.validate_score`, `by_bundle.setdefault(row['bundle_index'], []).append`, `by_bundle.setdefault` 외 9개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_page_mis_evidence` — L286**
```python
_page_mis_evidence(memberships, physical, counts, threshold, sums) -> tuple[dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dict`, `any`, `type`, `math.isfinite`, `ValueError`, `float`, `torch.tensor`, `math.fsum`, `sum`, `bundles.append` 외 3개
- 반환 표현: `(effective, evidence | {'sha256': _hash(evidence)})`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_page_mis_evidence` — L307**
```python
validate_page_mis_evidence(update: dict, protocol: dict, trees=None) -> dict[str, float]
```
- 기존 설명: Recompute shared ordinal MIS using all original masked page tokens.
- 직접 호출: `update.get`, `protocol.get`, `ValueError`, `pages.contract`, `type`, `set`, `any`, `tree.notebook.get`, `selected.items`, `_page_memberships` 외 12개
- 반환 표현: `physical`; `effective`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`effective_mis_deltas` — L345**
```python
effective_mis_deltas(update: dict, protocol: dict, *, trees=None) -> dict[str, float]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `validate_page_mis_evidence`
- 반환 표현: `validate_page_mis_evidence(update, protocol, trees)`

**`partition_learning_trees` — L349**
```python
partition_learning_trees(trees: list[ProblemTree], *, include_notebooks: bool=False, notebook_objective: dict | None=None) -> tuple[list[ProblemTree], list[str], list[str]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `has_learning_signal`, `tree.notebook.get('rewarded_notebook', {}).get`, `tree.notebook.get`, `audit.get`, `unresolved.append`, `eligible.append`, `signal_free.append`
- 반환 표현: `(eligible, unresolved, signal_free)`

**`action_mask` — L366**
```python
action_mask(sample: Sample, algorithm: str, tokenizer: Any) -> torch.Tensor
```
- 기존 설명: Only this completion's tokens; notebook-only spans use actual token decoding.
- 직접 호출: `torch.ones`, `len`, `mask.zero_`, `tokenizer.decode`, `ValueError`, `list`, `re.finditer`, `starts[0].start`, `re.search`, `starts[0].end` 외 2개
- 반환 표현: `mask`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`route_flags` — L395**
```python
route_flags(trees: list[ProblemTree], arm: str, seed: int, step: int) -> dict[str, bool]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `set`, `ValueError`, `tree.notebook.get('rewarded_notebook', {}).get`, `tree.notebook.get`, `audit.get`, `validate_update_binding`, `dict.fromkeys`, `sum`, `sorted` 외 3개
- 반환 표현: `dict.fromkeys(ids, False)`; `dict.fromkeys(ids, True)`; `{t.problem_id: t.route == 'VISION_ON' for t in trees}`; 외 1개 분기
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`effective_rewards` — L419**
```python
effective_rewards(groups: list[Group], algorithm: str, seed: int, step: int, *, notebook_reward_enabled: bool=True, notebook_scores: dict[str, dict] | None=None) -> dict[str, list[float]]
```
- 기존 설명: Control rewards do not overwrite observed outcomes or raw trajectories.
- 직접 호출: `type`, `ValueError`, `any`, `sum`, `len`, `hashlib.sha256(f'reward:{seed}:{step}:{sample.sample_id}'.encode()).digest`, `hashlib.sha256`, `f'reward:{seed}:{step}:{sample.sample_id}'.encode`, `float`, `random.Random(int.from_bytes(key, 'big')).random` 외 4개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `_PreparedAction` — L450**

기저: `object`.

```python
sample: Sample
request: RolloutRequest
mask: torch.Tensor
advantage: float
weight: float
norm: int
vision_on: bool
partial_credit: dict | None = None
old: torch.Tensor | None = None
reference: torch.Tensor | None = None
keep: bool = False
```

**`_retained_learning_signal` — L464**
```python
_retained_learning_signal(actions: list[_PreparedAction], clip: dict) -> dict
```
- 기존 설명: Policy coefficients at current=old; this is not measured model-gradient evidence.
- 직접 호출: `clip.get`, `torch.tensor`, `torch.tensor(action.advantage, device=action.old.device).to`, `torch.tensor([action.partial_credit['partial_token_advantages']], dtype=action.old.dtype,  …`, `ValueError`, `bool`, `torch.isfinite(coefficient).all`, `torch.isfinite`, `int`, `(active & action.mask.to(action.old.device)).sum` 외 2개
- 반환 표현: `{'required': True, 'basis': basis, 'policy_gradient_present': bool(tokens), 'retained_action_count': retained, 'nonzero_policy_gradient_tokens': tokens, 'signal_sample_ids': sample_ids}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_stage_time` — L499**
```python
_stage_time(device: torch.device) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.cuda.synchronize`, `time.monotonic`
- 반환 표현: `time.monotonic()`

**`_gradient_identity` — L506**
```python
_gradient_identity(parameters) -> dict
```
- 기존 설명: Distinguish an actual gradient contribution from allocating zero buffers.
- 직접 호출: `bool`, `torch.count_nonzero`, `tensor_sha256`, `parameters.items`
- 반환 표현: `{group: [None if p.grad is None or not bool(torch.count_nonzero(p.grad)) else tensor_sha256(p.grad) for p in values] for group, values in parameters.items()}`

**`_relationship_gradient_norms` — L514**
```python
_relationship_gradient_norms(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor, old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, loss_config: dict, partial: torch.Tensor, *, channel: str='RELATIONSHIP', accumulator=None) -> dict[str, float]
```
- 기존 설명: Measure one partial-credit VJP on the same forward without touching .grad.
- 직접 호출: `group_loss`, `torch.zeros`, `torch.tensor`, `bundle.params.items`, `torch.autograd.grad`, `dict.fromkeys`, `zip`, `bool`, `torch.isfinite(gradient).all`, `torch.isfinite` 외 9개
- 반환 표현: `norms`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`_relationship_gradient` — L541**
```python
_relationship_gradient(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor, old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, loss_config: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `torch.tensor`, `_relationship_gradient_norms`, `list`, `dict.fromkeys`, `norms.items`, `int`, `((partial > 0) & mask).sum`, `mask.sum`, `_hash`
- 반환 표현: `{'scope': 'relationship_policy_loss_only', 'measured': True, 'finite': True, 'credit_types': credit_types, 'positive_credit_types': positive_types, 'effect_scope': effect_scope, 'basis' …`

**`_native_relationship_vector` — L564**
```python
_native_relationship_vector(action: _PreparedAction, like: torch.Tensor, mask: torch.Tensor) -> tuple[list[float], torch.Tensor, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `torch.tensor`, `int`, `((partial > 0) & torch.tensor([positive], device=like.device) & mask).sum`
- 반환 표현: `(coefficients, partial, positive_tokens)`

**`_retained_native_relationship_signal` — L583**
```python
_retained_native_relationship_signal(actions: list[_PreparedAction]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_native_relationship_vector`, `action.mask.to`, `sample_ids.append`, `bool`
- 반환 표현: `{'required': True, 'basis': 'positive_native_coefficients_after_dtype_conversion_and_mis', 'policy_gradient_present': bool(tokens), 'positive_native_relationship_tokens': tokens, 'signa …`

**`_native_relationship_gradient_present` — L598**
```python
_native_relationship_gradient_present(gradient: dict, credit: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `gradient.get`, `_hash`, `set`, `all`, `math.isfinite`, `norms.values`, `any`
- 반환 표현: `gradient.get('scope') == 'native_error_relationship_policy_loss_only' and gradient.get('credit_type') == 'NATIVE_ERROR_RELATIONSHIP' and (gradient.get('measured') is True) and (gradient …`

**`_native_relationship_gradient` — L608**
```python
_native_relationship_gradient(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor, old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, loss_config: dict, relationship_gradient: dict | None=None) -> dict
```
- 기존 설명: Separate native credit from OUTPUT, facts, strict reward and KL after MIS.
- 직접 호출: `_native_relationship_vector`, `relationship_gradient.get`, `_hash`, `_relationship_gradient_norms`, `norms.items`, `int`, `((partial != 0) & mask).sum`
- 반환 표현: `receipt`; `receipt | {'measured': True, 'finite': True, 'credit_type': NATIVE_ERROR_RELATIONSHIP, 'basis': basis, 'nonzero_groups': [group for group, value in norms.items() if value > 0], 'group_n …`

**`_assessment_vector` — L635**
```python
_assessment_vector(action: _PreparedAction, like: torch.Tensor, mask: torch.Tensor) -> tuple[list[float], torch.Tensor, dict[str, int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `torch.tensor`, `bool`, `torch.isfinite(partial).all`, `torch.isfinite`, `ValueError`, `int`, `((partial > 0) & mask & torch.tensor([tokens], device=like.device)).sum`, `positive.items`
- 반환 표현: `(coefficients, partial, counts)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_retained_assessment_signal` — L653**
```python
_retained_assessment_signal(actions: list[_PreparedAction]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_assessment_vector`, `action.mask.to`, `sum`, `counts.values`, `sample_ids.append`, `counts.items`, `by_flags[flags]['signal_sample_ids'].append`, `bool`
- 반환 표현: `{'required': True, 'basis': 'positive_assessment_coefficients_after_dtype_conversion_and_mis', 'policy_gradient_present': bool(tokens), 'positive_assessment_tokens': tokens, 'signal_sam …`

**`_assessment_gradient_present` — L674**
```python
_assessment_gradient_present(gradient: dict, credit: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `gradient.get`, `_hash`, `set`, `all`, `math.isfinite`, `norms.values`, `any`
- 반환 표현: `gradient.get('scope') == 'native_assessment_policy_loss_only' and gradient.get('credit_type') == 'NATIVE_ASSESSMENT' and (gradient.get('measured') is True) and (gradient.get('finite') i …`

**`_assessment_gradient` — L684**
```python
_assessment_gradient(bundle: PolicyBundle, action: _PreparedAction, current: torch.Tensor, old: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, loss_config: dict, accumulator=None) -> dict
```
- 기존 설명: Exclude facts, relationships, strict reward and KL from assessment evidence.
- 직접 호출: `id`, `ValueError`, `_assessment_vector`, `sum`, `counts.values`, `bool`, `((partial != 0) & mask).any`, `_relationship_gradient_norms`, `norms.items`, `int` 외 2개
- 반환 표현: `receipt`; `receipt | {'measured': True, 'finite': True, 'credit_type': 'NATIVE_ASSESSMENT', 'basis': 'same_forward_assessment_only_vector_vjp', 'nonzero_groups': [group for group, value in norms.i …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `_AssessmentGradientAccumulator` — L706**

기저: `object`.

**`_AssessmentGradientAccumulator.__init__` — L708**
```python
_AssessmentGradientAccumulator.__init__(self, bundle: PolicyBundle) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `id`, `bundle.model.named_parameters`, `enumerate`, `len`, `ValueError`, `list`, `str`, `parameter.numel`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_AssessmentGradientAccumulator.validate` — L720**
```python
_AssessmentGradientAccumulator.validate(self, bundle: PolicyBundle) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_AssessmentGradientAccumulator`, `id`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_AssessmentGradientAccumulator.add` — L726**
```python
_AssessmentGradientAccumulator.add(self, parameter: torch.Tensor, gradient: torch.Tensor) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `id`, `ValueError`, `self.sums[identity].add_`, `gradient.detach().to`, `gradient.detach`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_AssessmentGradientAccumulator.receipt` — L735**
```python
_AssessmentGradientAccumulator.receipt(self, members: list[dict]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dict.fromkeys`, `self.sums.get`, `id`, `bool`, `torch.isfinite(value).all`, `torch.isfinite`, `RuntimeError`, `math.hypot`, `float`, `torch.linalg.vector_norm` 외 5개
- 반환 표현: `{'scope': 'signed_all_retained_native_assessment_policy_loss_only', 'basis': 'same_forward_vjp_sum_by_parameter_identity_cpu_float64', 'requirement': ASSESSMENT_REQUIREMENT, 'finite': T …`
- 명시적 예외: `RuntimeError`; 전파되는 callee 예외는 별도.

**`_assessment_member` — L754**
```python
_assessment_member(action: _PreparedAction) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_assessment_vector`, `action.mask.to`, `int`, `((partial != 0) & action.mask.to(action.old.device)).sum`, `_hash`, `action.mask.tolist`, `str`, `sum`, `counts.values`
- 반환 표현: `{'sample_id': action.sample.sample_id, 'problem_id': action.sample.problem_id, 'group_id': action.sample.group_id, 'policy_version': action.sample.policy_version, 'vision_on': action.vi …`

**`optimizer_step` — L770**
```python
optimizer_step(bundle: PolicyBundle, optimizer: Any, scheduler: Any, trees: list[ProblemTree], images: dict[str, list], preparation: Any, protocol: dict, algorithm: str, *, step: int, seed: int, vision_disabled: bool=False, notebook_reward_enabled: bool=True, require_retained_learning_signal: bool=False, observation_actors: dict[str, ActorItem] | None=None, objective_verifier: Any | None=None, on_progress: Callable[[dict[str, Any]], None] | None=None) -> dict
```
- 기존 설명: Compute all old/reference LPs first, then microbatch backward and ONE update.
- 직접 호출: `callable`, `ValueError`, `type`, `protocol.get`, `any`, `set`, `pages.contract`, `isinstance`, `objective.get`, `len` 외 117개
- 반환 표현: `base | {'status': 'NO_ACTIONS'}`; `base | {'status': 'NO_RETAINED_NATIVE_RELATIONSHIP_SIGNAL', 'seconds': timings}`; `base | {'status': 'NO_RETAINED_ASSESSMENT_SIGNAL', 'seconds': timings}`; 외 5개 분기
- 명시적 예외: `ValueError`, `RuntimeError`; 전파되는 callee 예외는 별도.

### D18.141 `fine_tuning/notebook_rl/vault.py`

[현재 소스](../../fine_tuning/notebook_rl/vault.py) · SHA256 `538ac47c3465f9269c21b75a28746d49656477274bf2d363d623a56f5cc1fa44`

모듈의 기존 설명: Write-once split records; actor reads never open verifier or lineage payloads.

**`_sha` — L33**
```python
_sha(raw: bytes) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(raw).hexdigest`, `hashlib.sha256`
- 반환 표현: `hashlib.sha256(raw).hexdigest()`

**`_json` — L37**
```python
_json(value: object) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ': …`, `json.dumps`
- 반환 표현: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')`

**`_ids` — L42**
```python
_ids(ids: set[str]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `_json`, `sorted`
- 반환 표현: `_sha(_json(sorted(ids)))`

**`_safe_root` — L46**
```python
_safe_root(root: Path, *, create: bool=False) -> Path
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `ValueError`, `root.absolute`, `reversed`, `path.is_symlink`, `path.exists`, `path.is_dir`, `root.mkdir`, `root.is_dir`
- 반환 표현: `root`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_read` — L63**
```python
_read(path: Path) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.lstat`, `stat.S_ISREG`, `ValueError`, `os.open`, `os.fdopen`, `os.fstat`, `stream.fileno`, `stream.read`
- 반환 표현: `stream.read()`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_code_hashes` — L78**
```python
_code_hashes() -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_sha`, `(_HERE / name).read_bytes`
- 반환 표현: `{name: _sha((_HERE / name).read_bytes()) for name in _CODE}`

**`_unique` — L82**
```python
_unique(pairs: list[tuple[str, object]]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_manifest` — L91**
```python
_manifest(root: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `_read`, `ValueError`, `type`, `set`, `isinstance`, `_SHA.fullmatch`, `data['files'].values`, `len`, `_code_hashes` 외 1개
- 반환 표현: `data`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_check_lineage` — L129**
```python
_check_lineage(lineage: Lineage) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_load` — L134**
```python
_load(root: Path, name: str, cls: type[T]) -> list[T]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_safe_root`, `_manifest`, `_read`, `_sha`, `ValueError`, `raw.decode`, `raw.endswith`, `text.split`, `from_json`, `len` 외 3개
- 반환 표현: `rows`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`write_split` — L160**
```python
write_split(root: Path, actors: list[ActorItem], verifiers: list[VerifierItem], lineages: list[Lineage]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `zip`, `any`, `type`, `ValueError`, `b''.join`, `to_json(row).encode`, `to_json`, `len`, `id_sets.append`, `_check_lineage` 외 21개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`; 전파되는 callee 예외는 별도.

**`read_actors` — L214**
```python
read_actors(root: Path) -> list[ActorItem]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_load`
- 반환 표현: `_load(root, 'actor_inputs.jsonl', ActorItem)`

**`read_verifier` — L218**
```python
read_verifier(root: Path) -> dict[str, VerifierItem]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_load`
- 반환 표현: `{row.problem_id: row for row in _load(root, 'verifier_private.jsonl', VerifierItem)}`

**`read_lineage` — L222**
```python
read_lineage(root: Path) -> dict[str, Lineage]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_load`
- 반환 표현: `{row.problem_id: row for row in _load(root, 'lineage.jsonl', Lineage)}`

### D18.142 `fine_tuning/notebook_rl/verifier.py`

[현재 소스](../../fine_tuning/notebook_rl/verifier.py) · SHA256 `772deabbc4c3536273296954b1fabbc8181d3e38d5bc289ce7319114f95490d7`

모듈의 기존 설명: Private deterministic rewards. This object must stay in the verifier process.

**`diagnosis_fields` — L15**
```python
diagnosis_fields(text: str) -> tuple[str, str] | None
```
- 기존 설명: A single three-line note, not a search for whichever label earns reward.
- 직접 호출: `re.fullmatch`, `match[2].strip`, `match[3].strip`, `match[1].startswith`, `slot_id`
- 반환 표현: `None`; `('VISUAL' if match[1].startswith('VISUAL_') else match[1], slot_id(match[2]))`

**클래스 `Verifier` — L25**

기저: `object`.

**`Verifier.__init__` — L26**
```python
Verifier.__init__(self, vault_root: Path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read_verifier`, `read_actors`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`Verifier.trace` — L30**
```python
Verifier.trace(self, problem_id: str, text: str, finish_reason: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `check_trace`
- 반환 표현: `check_trace(actor=self.actors[problem_id], item=self.items[problem_id], text=text, finish_reason=finish_reason)`

**`Verifier.cause` — L35**
```python
Verifier.cause(self, problem_id: str, result: VerifierResult) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `attempt_cause`
- 반환 표현: `attempt_cause(result, self.items[problem_id])`

**`Verifier.diagnosis_target` — L39**
```python
Verifier.diagnosis_target(self, problem_id: str, result: VerifierResult) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `diagnosis_target`
- 반환 표현: `diagnosis_target(result, self.items[problem_id])`

**`Verifier.route` — L43**
```python
Verifier.route(self, problem_id: str, results: list[VerifierResult], rule: str) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `problem_route`
- 반환 표현: `problem_route(results, self.items[problem_id], rule)`

**`Verifier.grade` — L47**
```python
Verifier.grade(self, problem_id: str, text: str, finish_reason: str, mode: Mode) -> VerifierResult
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `extract_final`, `compare_slots`, `parse_observations`, `VerifierResult`, `is_ambiguous`, `compare`, `float`, `to_json`
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`Verifier.grade_diagnosis` — L66**
```python
Verifier.grade_diagnosis(self, problem_id: str, text: str, target: DiagTarget) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `slot_id`, `set`, `native_absent_relations`, `ValueError`, `diagnosis_fields`
- 반환 표현: `0.0`; `1.0 + 0.25 * (target.cause3 == 'VISUAL' and fields[1] in {slot_id(k) for k in target.wrong_slots})`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`answer_reward` — L81**
```python
answer_reward(result: VerifierResult, *, visual_lambda: float=0.0) -> float
```
- 기존 설명: Optional S4 bonus is bounded and cannot reward facts-only or invalid answers.
- 직접 호출: `to_json`, `type`, `ValueError`
- 반환 표현: `result.reward_answer`; `result.reward_answer + visual_lambda * result.slots.required_correct / result.slots.required`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.143 `fine_tuning/notebook_rl/verifier_service.py`

[현재 소스](../../fine_tuning/notebook_rl/verifier_service.py) · SHA256 `482a0d3af61f572642b3b36dee2573a095ef9829495165ddca72879d53669f1d`

모듈의 기존 설명: Fresh-process verifier facade; only bounded JSON metadata crosses the pipes.

**클래스 `VerifierServiceError` — L46**

기저: `RuntimeError`.

**클래스 `_Rejected` — L50**

기저: `Exception`.

**`_wire` — L54**
```python
_wire(value: object) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ': …`, `json.dumps`
- 반환 표현: `json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')`

**`_unique` — L59**
```python
_unique(pairs) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_decode` — L68**
```python
_decode(raw: bytes) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`
- 반환 표현: `json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid)`

**`_check_text` — L74**
```python
_check_text(value: object, *, identity: bool=False) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `value.strip`, `len`, `_Rejected`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `_Rejected`; 전파되는 callee 예외는 별도.

**`_record` — L79**
```python
_record(cls, value) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `_Rejected`, `from_json`, `_wire(value).decode`, `_wire`
- 반환 표현: `from_json(cls, _wire(value).decode())`
- 명시적 예외: `_Rejected`; 전파되는 callee 예외는 별도.

**`_slot_names` — L88**
```python
_slot_names(names) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `any`, `isinstance`, `_Rejected`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `_Rejected`; 전파되는 callee 예외는 별도.

**`_result` — L94**
```python
_result(value: VerifierResult) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `to_json`, `_Rejected`, `_slot_names`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `_Rejected`; 전파되는 callee 예외는 별도.

**`_route` — L103**
```python
_route(value, size: int) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `len`, `_Rejected`, `set`, `_slot_names`
- 반환 표현: `(route, data)`
- 명시적 예외: `_Rejected`; 전파되는 callee 예외는 별도.

**클래스 `_Issued` — L130**

기저: `object`.

**`_Issued.__init__` — L131**
```python
_Issued.__init__(self) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `OrderedDict`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_Issued.key` — L134**
```python
_Issued.key(self, kind, pid, record) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(to_json(record).encode()).hexdigest`, `hashlib.sha256`, `to_json(record).encode`, `to_json`
- 반환 표현: `(kind, pid, hashlib.sha256(to_json(record).encode()).hexdigest())`

**`_Issued.add` — L137**
```python
_Issued.add(self, kind, pid, record) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.key`, `self.values.move_to_end`, `len`, `self.values.popitem`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`_Issued.require` — L144**
```python
_Issued.require(self, kind, pid, record) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.key`, `_Rejected`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `_Rejected`; 전파되는 callee 예외는 별도.

**`_dispatch` — L149**
```python
_dispatch(verifier, issued: _Issued, request: dict) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `set`, `_Rejected`, `_check_text`, `verifier.trace`, `validate_trace_report`, `verifier.grade`, `_result`, `issued.add`, `json.loads` 외 11개
- 반환 표현: `report`; `json.loads(to_json(result))`; `value`; 외 4개 분기
- 명시적 예외: `_Rejected`; 전파되는 callee 예외는 별도.

**`_worker_read` — L211**
```python
_worker_read(stream, size: int) -> bytes
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bytearray`, `len`, `stream.read`, `chunks.extend`, `bytes`
- 반환 표현: `bytes(chunks)`
- 명시적 예외: `EOFError`; 전파되는 callee 예외는 별도.

**`_send` — L221**
```python
_send(stream, payload: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_wire`, `len`, `_Rejected`, `stream.write`, `struct.pack`, `stream.flush`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `_Rejected`; 전파되는 callee 예외는 별도.

**`_worker` — L229**
```python
_worker(vault_root: Path, forbidden_roots: list[Path]) -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `install_guards`, `vault._safe_root`, `hashlib.sha256(vault._read(vault_root / 'manifest.yaml')).hexdigest`, `hashlib.sha256`, `vault._read`, `Verifier`, `list`, `verifier.actors.values`, `vault.read_lineage`, `set` 외 17개
- 반환 표현: `2`; `3`; `0`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_stop_process` — L290**
```python
_stop_process(process: subprocess.Popen) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `stream.close`, `process.poll`, `process.wait`, `os.killpg`, `process.terminate`, `process.kill`
- 반환 표현: `None`

**클래스 `VerifierService` — L316**

기저: `object`.

**`VerifierService.__init__` — L318**
```python
VerifierService.__init__(self, vault_root: Path, *, forbidden_roots: list[Path], timeout: float=30.0, max_request_bytes: int=MAX_FRAME_BYTES) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `math.isfinite`, `any`, `isinstance`, `ValueError`, `float`, `threading.Lock`, `str`, `Path(vault_root).absolute`, `Path` 외 20개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`, `VerifierServiceError`, `re-raise`; 전파되는 callee 예외는 별도.

**`VerifierService.metadata` — L367**
```python
VerifierService.metadata(self) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_decode`, `_wire`
- 반환 표현: `_decode(_wire(self._metadata))`

**`VerifierService._io` — L370**
```python
VerifierService._io(self, descriptor: int, data: bytes | int, deadline: float) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `memoryview`, `len`, `bytearray`, `selectors.DefaultSelector`, `selector.register`, `time.monotonic`, `selector.select`, `TimeoutError`, `os.read` 외 4개
- 반환 표현: `bytes(pending) if reading else None`
- 명시적 예외: `TimeoutError`, `VerifierServiceError`; 전파되는 callee 예외는 별도.

**`VerifierService._response` — L397**
```python
VerifierService._response(self, request_id: int, deadline: float) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._process.stdout.fileno`, `struct.unpack`, `self._io`, `VerifierServiceError`, `_decode`, `type`, `value.get`, `set`, `isinstance`, `_Rejected`
- 반환 표현: `value['result']`
- 명시적 예외: `VerifierServiceError`, `_Rejected`; 전파되는 callee 예외는 별도.

**`VerifierService._request` — L415**
```python
VerifierService._request(self, method: str, problem_id: str, params: dict) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `VerifierServiceError`, `_wire`, `len`, `ValueError`, `time.monotonic`, `self._io`, `self._process.stdin.fileno`, `struct.pack`, `self._response`, `str` 외 1개
- 반환 표현: `self._response(self._sequence, deadline)`
- 명시적 예외: `VerifierServiceError`, `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`VerifierService.grade` — L434**
```python
VerifierService.grade(self, problem_id: str, text: str, finish_reason: str, mode: str) -> VerifierResult
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._request`, `_record`, `_result`, `_Rejected`, `self.close`, `VerifierServiceError`
- 반환 표현: `result`
- 명시적 예외: `_Rejected`, `VerifierServiceError`; 전파되는 callee 예외는 별도.

**`VerifierService.trace` — L446**
```python
VerifierService.trace(self, problem_id: str, text: str, finish_reason: str) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._request`, `validate_trace_report`, `self.close`, `VerifierServiceError`
- 반환 표현: `report`
- 명시적 예외: `VerifierServiceError`; 전파되는 callee 예외는 별도.

**`VerifierService.cause` — L455**
```python
VerifierService.cause(self, problem_id: str, result: VerifierResult) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._request`, `json.loads`, `to_json`, `isinstance`, `self.close`, `VerifierServiceError`
- 반환 표현: `value`
- 명시적 예외: `VerifierServiceError`; 전파되는 callee 예외는 별도.

**`VerifierService.diagnosis_target` — L462**
```python
VerifierService.diagnosis_target(self, problem_id: str, result: VerifierResult) -> DiagTarget | None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._request`, `json.loads`, `to_json`, `_record`, `_slot_names`, `self.close`, `VerifierServiceError`
- 반환 표현: `target`
- 명시적 예외: `VerifierServiceError`; 전파되는 callee 예외는 별도.

**`VerifierService.route` — L473**
```python
VerifierService.route(self, problem_id: str, results: list[VerifierResult], rule: str) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `len`, `ValueError`, `self._request`, `json.loads`, `to_json`, `_route`, `self.close`, `VerifierServiceError`
- 반환 표현: `_route(value, len(results))`
- 명시적 예외: `ValueError`, `VerifierServiceError`; 전파되는 callee 예외는 별도.

**`VerifierService.grade_diagnosis` — L483**
```python
VerifierService.grade_diagnosis(self, problem_id: str, text: str, target: DiagTarget) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._request`, `json.loads`, `to_json`, `type`, `self.close`, `VerifierServiceError`, `float`
- 반환 표현: `float(value)`
- 명시적 예외: `VerifierServiceError`; 전파되는 callee 예외는 별도.

**`VerifierService.close` — L490**
```python
VerifierService.close(self) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self._finalizer`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`VerifierService.__enter__` — L494**
```python
VerifierService.__enter__(self) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `self`

**`VerifierService.__exit__` — L497**
```python
VerifierService.__exit__(self, *_) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `self.close`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`main` — L501**
```python
main() -> int
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `_worker`
- 반환 표현: `_worker(args.vault_root, args.forbidden_root)`

### D18.144 `fine_tuning/notebook_rl/whole_reference_credit.py`

[현재 소스](../../fine_tuning/notebook_rl/whole_reference_credit.py) · SHA256 `1d6801be60f74cab0503bdd0d8dc5d091f389055719771938721798e7909f6a7`

모듈의 기존 설명: Diagnostic unit checks on rendered references, with credit on original raw tokens.

**`_seal` — L28**
```python
_seal(row: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `digest`
- 반환 표현: `row | {'score_sha256': digest(row)}`

**`_binding` — L32**
```python
_binding(sample) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `any`, `ValueError`, `digest`, `hashlib.sha256(sample.text.encode()).hexdigest`, `hashlib.sha256`, `sample.text.encode`
- 반환 표현: `{'sample_id': sample.sample_id, 'group_id': sample.group_id, 'text_sha256': digest(sample.text), 'raw_text_sha256': hashlib.sha256(sample.text.encode()).hexdigest(), 'token_ids_sha256': …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`check` — L44**
```python
check(sample, tree, audit: dict, actor, verifier) -> dict
```
- 기존 설명: Freshly grade the rendering and independent units; retain only raw spans.
- 직접 호출: `_binding`, `synthesis.audit_episode`, `type`, `_hash`, `ValueError`, `episode.solution_samples`, `reference.expand`, `reference.validate_audit_coverage`, `synthesis.grade_note`, `verifier.grade` 외 19개
- 반환 표현: `_seal({'version': VERSION, 'diagnostic_only': True, 'generation_policy': reference.VERSION, 'expansion_status': expanded['status'], 'expansion_captured': captured, 'expanded_text_sha256 …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_raw_decisions` — L120**
```python
_raw_decisions(text: str) -> dict[str, tuple[Any, list[int]]]
```
- 기존 설명: Walk JSON offsets, so identical decision literals retain their own position.
- 직접 호출: `reference._decode`, `len`, `set`, `ValueError`, `enumerate`
- 반환 표현: `{row[0]: (row[3], spans['assessments', index, 3]) for index, row in enumerate(note['assessments']) if row[3] is not None}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_score` — L130**
```python
validate_score(row: dict, sample) -> None
```
- 기존 설명: Validate raw bookkeeping; semantic claims still require a fresh check().
- 직접 호출: `type`, `row.get`, `digest`, `row.items`, `any`, `_binding(sample).items`, `_binding`, `existing._strict_digest`, `row['strict_report'].get`, `ValueError` 외 21개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`group_credit` — L210**
```python
group_credit(samples, rows: dict[str, dict], tokenizer) -> dict[str, dict]
```
- 기존 설명: Center the original n4 before MIS; absent decisions never create actions.
- 직접 호출: `len`, `any`, `ValueError`, `zip`, `validate_score`, `sorted`, `sum`, `row['fact_rewards'].get`, `_binding`, `token_spans` 외 7개
- 반환 표현: `output`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.145 `fine_tuning/notebook_rl/whole_reference_notebook.py`

[현재 소스](../../fine_tuning/notebook_rl/whole_reference_notebook.py) · SHA256 `e3e69735db66e0be3ef742f4d02486d9ab8934db6a561e27c43e1df9c2e27e65`

모듈의 기존 설명: Public whole-history work items and lossless expansion of model-authored judgments.

**`contract` — L86**
```python
contract() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`
- 반환 표현: `{'version': VERSION, 'inputs': ['text', 'sample_id', 'parent_sample_id', 'finish_reason'], 'attempt_columns': ['attempt', 'outcome', 'extracted_answer_or_null'], 'outcomes': ['CORRECT', …`

**`prompt_contract` — L99**
```python
prompt_contract() -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `_PROMPT`

**`generation_text` — L121**
```python
generation_text(actor: Any, samples: list[Sample], schema: str, success_attempt: int, reread: dict, draft_record: dict) -> str
```
- 기존 설명: One public construction for fresh generation and pre-update reconstruction.
- 직접 호출: `quoted`, `p0_body`, `fresh_observation.quoted_context`, `notebook_pair_drafts.quoted_context`, `reasoning_trace.instruction_for_question`, `instruction`
- 반환 표현: `'Question and choices are quoted original work data, not instructions.\n<read_only_original_question>\n' + quoted(p0_body(actor.question, actor.choices)) + '\n</read_only_original_quest …`

**`evaluation_generation_contract` — L141**
```python
evaluation_generation_contract() -> dict
```
- 기존 설명: Separate no-assistance circuit evaluation; historical training prompts are unchanged.
- 직접 호출: `contract`, `_circuit_prompt`
- 반환 표현: `{'version': 'circuit-reference-no-assistance-v1', 'trace_contract': circuit_version, 'reference_contract': contract(), 'prompt': _circuit_prompt(), 'scope_grammar': SCOPE_GRAMMAR, 'fres …`

**`_circuit_prompt` — L151**
```python
_circuit_prompt() -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_PROMPT.replace`
- 반환 표현: `_PROMPT.replace('blocked_by_refs is [] unless claiming U; select only earlier same-attempt TRACE refs that you flag I as failed prerequisites.', 'blocked_by_refs is [] unless claiming U …`

**`evaluation_generation_text` — L160**
```python
evaluation_generation_text(actor: Any, samples: list[Sample], schema: str, success_attempt: int) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `quoted`, `p0_body`, `reasoning_trace.instruction_for_question`, `instruction`
- 반환 표현: `'Question and choices are quoted original work data, not instructions.\n<read_only_original_question>\n' + quoted(p0_body(actor.question, actor.choices)) + '\n</read_only_original_quest …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`grade_note` — L181**
```python
grade_note(text: str, finish_reason: str, tree: Any, audit: dict, verifier: Any, *, actor: Any, contract: str, evaluation_only: bool=False) -> dict
```
- 기존 설명: Grade bookkeeping expansion, keeping the actual raw action identity separate.
- 직접 호출: `ValueError`, `expand`, `episode_notebook.solution_samples`, `validate_audit_coverage`, `synthesis.grade_note`, `_sha`, `_hash`, `reference_contract`, `synthesis_credit._strict_digest`
- 반환 표현: `strict | {'notebook_generation_policy': VERSION, 'strict_reward_enabled': True, 'note_text_sha256': _sha(text), 'note_finish_reason': finish_reason, 'expansion_status': expanded['status …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_sha` — L205**
```python
_sha(text: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(text.encode('utf-8')).hexdigest`, `hashlib.sha256`, `text.encode`
- 반환 표현: `hashlib.sha256(text.encode('utf-8')).hexdigest()`

**`public_inventory` — L209**
```python
public_inventory(samples: list[Sample]) -> dict
```
- 기존 설명: Read public text, identity, ancestry and termination; retain every raw occurrence.
- 직접 호출: `ValueError`, `set`, `type`, `seen.add`, `public_coverage`, `enumerate`, `_first_attempt_lines`, `len`, `attempts.append`, `_sha` 외 16개
- 반환 표현: `{**result, 'sha256': _hash(result)}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`instruction` — L276**
```python
instruction(samples: list[Sample], schema: str, designated_success_attempt: int, *, trace_contract: str | None=None) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `_circuit_prompt`, `public_inventory`, `type`, `len`, `_first_attempt_lines`, `enumerate`, `inventory['references'].items`, `template.format`, `quoted`
- 반환 표현: `template.format(schema=schema, success_attempt=designated_success_attempt, inventory=quoted(wire), transcript=quoted(transcript))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_audit_coverage` — L310**
```python
validate_audit_coverage(inventory: dict, audit: dict) -> dict
```
- 기존 설명: Host-only guard: private checks must be expressible before any generation.
- 직접 호출: `inventory.get`, `_hash`, `inventory.items`, `ValueError`, `audit.get`, `type`, `len`, `inventory['references'].values`, `zip`, `any` 외 1개
- 반환 표현: `{'status': 'AUDIT_COVERAGE_COMPLETE', 'checks': count, 'sha256': inventory['sha256']}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_decode` — L331**
```python
_decode(text: str) -> tuple[Any, dict[tuple, list[int]]]
```
- 기존 설명: Use JSON decoder boundaries, including escapes; never search repeated literals.
- 직접 호출: `json.JSONDecoder`, `decoder.decode`, `visit`, `ws`
- 반환 표현: `(value, spans)`

**`_answer_shape` — L363**
```python
_answer_shape(value: Any, *, nullable: bool=False) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `_single_line_scalar`, `value.is_finite`
- 반환 표현: `value is None and nullable or (type(value) is str and _single_line_scalar(value)) or type(value) is int or (type(value) is Decimal and value.is_finite())`

**`_endpoint` — L370**
```python
_endpoint(row: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `copy.deepcopy`
- 반환 표현: `{key: copy.deepcopy(row[key]) for key in ('attempt', 'kind', 'rule', 'locus', 'evidence_ranges')}`

**`expand` — L374**
```python
expand(text: str, finish_reason: str, samples: list[Sample]) -> dict
```
- 기존 설명: Fail closed on compact shape errors; successful expansion remains unverified.
- 직접 호출: `public_inventory`, `_decode`, `type`, `set`, `len`, `_answer_shape`, `ValueError`, `note['corrected_facts'].items`, `_single_line_scalar`, `_native_value_shape` 외 18개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.146 `fine_tuning/notebook_rl/whole_reference_page_rollout.py`

[현재 소스](../../fine_tuning/notebook_rl/whole_reference_page_rollout.py) · SHA256 `05868037a5ef5dd606696179404d92bccb797901b28c2c3e65d30be2732a6ff1`

모듈의 기존 설명: Collect real n4 page actions; assemble four ordinal notebooks only for grading.

**`generation_messages` — L18**
```python
generation_messages(actor, originals, schema, success_attempt, page_index, reread, draft_record) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `reference.generation_text`, `reference.instruction`, `original.endswith`, `ValueError`, `drafts.quoted_context`, `draft_context.partition`, `len`, `wrapper.count`, `context.count`, `wrapper.replace` 외 3개
- 반환 표현: `[user_message(body, len(actor.image_paths))]`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_request` — L34**
```python
_request(tree, actor, schema, page, reread, draft_record, images) -> RolloutRequest
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `RolloutRequest`, `generation_messages`, `episode.solution_samples`, `seeds`
- 반환 표현: `RolloutRequest(f'{tree.problem_id}:v{tree.policy_version}:{branch}', None, generation_messages(actor, episode.solution_samples(tree), schema, tree.notebook['rewarded_notebook']['episode …`

**`_group_hash` — L44**
```python
_group_hash(tree) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `_hash`, `json.loads`, `to_json`
- 반환 표현: `_hash([json.loads(to_json(group)) for group in tree.groups if group.kind == 'DIAG'])`

**`_call` — L48**
```python
_call(request, page, actor, prepared) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `copy.deepcopy`, `_hash`, `json.loads`, `json.dumps`, `_image_identity`, `list`, `range`
- 반환 표현: `{'page_index': page['page_index'], 'group_id': request.request_id, 'messages': copy.deepcopy(request.messages), 'messages_sha256': _hash(request.messages), 'images': json.loads(json.dum …`

**`reconstruct_bundles` — L57**
```python
reconstruct_bundles(tree, audit, verifier, *, actor) -> list[dict]
```
- 기존 설명: Fresh strict grades use the assembled text without constructing sampled actions.
- 직접 호출: `episode.solution_samples`, `len`, `pages.plan`, `any`, `ValueError`, `range`, `pages.assemble`, `reference.grade_note`, `_hash`, `json.loads` 외 2개
- 반환 표현: `bundles`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_physical_grade` — L78**
```python
_physical_grade(sample, page_index, bundle) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `reference._sha`
- 반환 표현: `bundle['grade'] | {'sample_id': sample.sample_id, 'text': sample.text, 'eligible_for_rl_loss': True, 'notebook_generation_policy': pages.VERSION, 'page_index': page_index, 'bundle_ordin …`

**`_collect` — L86**
```python
_collect(tree, actor, images, engine, verifier, cfg, *, on_page, should_stop) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `should_stop`, `record.update`, `_request`, `engine.prepare`, `len`, `ValueError`, `str(exc).startswith`, `str`, `_call`, `_sample_request` 외 20개
- 반환 표현: `None`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`append` — L156**
```python
append(trees, images, engine, verifier, prompts, cfg, *, actors, on_group=None, on_observation=None, on_draft=None, on_page=None, should_stop=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `cfg.validate`, `pages.contract`, `_hash`, `reference.contract`, `pages.prompt_contract`, `drafts.contract`, `len`, `set`, `prompts.get`, `any` 외 26개
- 반환 표현: `report`
- 명시적 예외: `ValueError`, `re-raise`; 전파되는 callee 예외는 별도.

**`validate` — L241**
```python
validate(tree, *, tokenizer=None, preparation=None, images=None, observation_actor=None, observation_schema=None, observation_master_seed=None, **policies) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `tree.notebook.get`, `any`, `policies.items`, `ValueError`, `pages.contract`, `_hash`, `reference.contract`, `drafts.contract`, `note.get`, `from_json` 외 35개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`regrade` — L371**
```python
regrade(tree: ProblemTree, actor: ActorItem, verifier: Any, tokenizer: Any) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `validate`, `episode.audit_episode`, `synthesis.audit_episode`, `reconstruct_bundles`, `_hash`, `row.items`, `tree.notebook.get`
- 반환 표현: `receipt | {'sha256': _hash(receipt)}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.147 `fine_tuning/notebook_rl/whole_reference_pages.py`

[현재 소스](../../fine_tuning/notebook_rl/whole_reference_pages.py) · SHA256 `bb1ae19fdc3e456a8c26cbe64b720b52af64bfd5b1ad41a1d698b079bcebf61a`

모듈의 기존 설명: Bounded public notebook pages, assembled only for independent verification.

**`prompt_contract` — L57**
```python
prompt_contract() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `reference.prompt_contract`
- 반환 표현: `{'version': VERSION, 'reference': reference.prompt_contract(), 'scope': reference.SCOPE_GRAMMAR, 'intro': _INTRO, 'header_grammar': _HEADER_GRAMMAR, 'header_example': _HEADER_EXAMPLE, ' …`

**`contract` — L67**
```python
contract() -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `reference.contract`, `list`
- 반환 표현: `{'version': VERSION, 'reference_contract': reference.contract(), 'group_size': 4, 'total_tokens': 8192, 'header_tokens': 2048, 'assessment_tokens': 6144, 'allocation': 'proportional-ref …`

**`plan` — L80**
```python
plan(originals: list[Sample], total_tokens: int=8192) -> dict
```
- 기존 설명: Allocate the fixed output budget using only the public reference inventory.
- 직접 호출: `type`, `ValueError`, `reference.public_inventory`, `range`, `len`, `sum`, `sorted`, `any`, `pages.extend`, `enumerate` 외 3개
- 반환 표현: `{**result, 'sha256': _hash(result)}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`instruction` — L107**
```python
instruction(originals: list[Sample], schema: str, success_attempt: int, page_index: int) -> str
```
- 기존 설명: Keep every source and destination visible; request only this page's claims.
- 직접 호출: `plan`, `type`, `len`, `ValueError`, `json.dumps(schema, ensure_ascii=False).replace('<', '\\u003c').replace`, `json.dumps(schema, ensure_ascii=False).replace`, `json.dumps`, `reference.instruction`, `prompt.partition`, `readonly.endswith` 외 5개
- 반환 표현: `reference.SCOPE_GRAMMAR + '\n' + _INTRO + ownership + '\nReturn one bare JSON object.\n' + grammar + _READONLY_GRAMMAR + readonly + _ENDING + ownership + _TAIL`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_bindings` — L133**
```python
_bindings(page_samples: list[Sample], originals: list[Sample], page_count: int) -> list[dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `len`, `ValueError`, `any`, `set`, `enumerate`, `isinstance`, `ids.add`, `groups.add`, `bindings.append` 외 1개
- 반환 표현: `bindings`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`assemble` — L161**
```python
assemble(page_samples: list[Sample], originals: list[Sample]) -> dict
```
- 기존 설명: Join one ordinal's original pages, preserving every copied JSON literal.
- 직접 호출: `plan`, `_bindings`, `len`, `result.update`, `any`, `enumerate`, `reference._decode`, `set`, `type`, `ValueError` 외 13개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.148 `fine_tuning/nvw.py`

[현재 소스](../../fine_tuning/nvw.py) · SHA256 `bfb2913517fd72b37d39f2ea8ad502a96e4fee82301865b586c17365c280b3e9`

모듈의 기존 설명: Private independent synthetic visual worlds. No benchmark or cloud inputs.

**`digest` — L28**
```python
digest(value: Any) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdiges …`, `hashlib.sha256`, `json.dumps(value, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`seeds` — L32**
```python
seeds(master: int, index: int, family: str) -> dict[str, int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `digest`
- 반환 표현: `{key: int(digest([VERSION, master, index, family, key])[:16], 16) for key in ('world', 'structure', 'rule', 'support', 'glyph', 'layout', 'style', 'options')}`

**`family_split` — L37**
```python
family_split(family_id: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`
- 반환 표현: `'train' if bucket < 8 else 'development' if bucket == 8 else 'sealed_holdout'`

**`solve_signal` — L42**
```python
solve_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `range`, `set`, `states.append`
- 반환 표현: `states`

**`verify_signal` — L50**
```python
verify_signal(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `range`, `result.append`, `sorted`, `len`, `list`
- 반환 표현: `result`

**`spatial_predicate` — L61**
```python
spatial_predicate(obj: dict[str, Any], pred: list[Any], objects: list[dict[str, Any]]) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `obj['type'] == value`; `obj['color'] == value`; `obj['x'] < 2 if value == 'L' else obj['x'] == 2`; 외 1개 분기

**`solve_spatial` — L73**
```python
solve_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `spatial_predicate`, `sorted`
- 반환 표현: `[sorted(sets[0]), sorted(sets[1]), sorted(combined)]`

**`verify_spatial` — L80**
```python
verify_spatial(world: dict[str, Any]) -> list[list[int]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `next`, `masks.append`, `range`, `len`
- 반환 표현: `[[i for i in range(len(world['objects'])) if value & 1 << i] for value in [*masks, joined]]`

**`solve_fsm` — L101**
```python
solve_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `states.append`
- 반환 표현: `states`

**`verify_fsm` — L108**
```python
verify_fsm(mapping: list[str], sequence: list[int], initial: int) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bool`, `int`, `ValueError`, `states.append`
- 반환 표현: `states`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`fsm_identifiability` — L125**
```python
fsm_identifiability(world: dict[str, Any]) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `list`, `itertools.product`, `all`, `ValueError`, `verify_fsm`, `sorted`, `range`, `len`
- 반환 표현: `{'answer_identifiable': len(answers) == 1, 'mapping_identifiable': len(candidates) == 1, 'rule_identifiable': len(candidates) == 1, 'state_prefix_identifiable': [len(values) == 1 for va …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_world` — L142**
```python
make_world(family: str, streams: dict[str, int], index: int) -> tuple[dict[str, Any], str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `random.Random`, `rng.randrange`, `range`, `rng.random`, `rules.randrange`, `rules.choice`, `rules.sample`, `ValueError`, `list`, `enumerate`
- 반환 표현: `(world, 'NVW-K' if index % 2 else 'NVW-S')`; `({'objects': objects, 'filters': rules.sample(predicates, 2), 'operation': rules.choice(['UNION', 'INTERSECTION'])}, 'NVW-K' if index % 2 else 'NVW-S')`; `({'mapping': mapping, 'sequence': [rng.randrange(2) for _ in range(rng.randrange(2, 5))], 'initial': rng.randrange(2), 'support': support}, track)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`semantic_family` — L173**
```python
semantic_family(world: dict[str, Any], family: str) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `min`, `list`, `reversed`, `range`, `tuple`, `sorted`, `enumerate`, `set`, `itertools.product`, `itertools.permutations` 외 3개
- 반환 표현: `family + '-' + digest(key)`

**`glyph` — L206**
```python
glyph(draw: ImageDraw.ImageDraw, x: int, y: int, code: int, radius: int=24) -> None
```
- 기존 설명: Distinct procedural polygons with a four-bit central marker; no external assets.
- 직접 호출: `int`, `math.cos`, `math.sin`, `range`, `draw.polygon`, `draw.rectangle`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`text` — L219**
```python
text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, small: bool=False) -> list[int]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `draw.textbbox`, `ValueError`, `draw.text`
- 반환 표현: `bbox`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`arrow` — L228**
```python
arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.hypot`, `draw.line`, `draw.polygon`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`filter_name` — L239**
```python
filter_name(pred: list[Any]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: `{'type': f'glyph type {value}', 'color': f'color {value}', 'inside': f'inside region {value}', 'left_of': f'left of object {value}', 'above': f'above object {value}'}[kind]`

**`render` — L246**
```python
render(episode: dict[str, Any], canonical: bool=False) -> tuple[Image.Image, dict[str, Any]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.new`, `ImageDraw.Draw`, `random.Random`, `text`, `family.replace`, `draw.rectangle`, `range`, `glyph`, `'; '.join`, `legend.append` 외 14개
- 반환 표현: `(image, observation)`

**`build_episode` — L350**
```python
build_episode(family: str, index: int, master_seed: int=3407) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `seeds`, `make_world`, `solve_fsm`, `verify_fsm`, `fsm_identifiability`, `solver`, `verifier`, `len`, `random.Random(streams['options']).sample`, `random.Random` 외 15개
- 반환 표현: `{'id': f'{VERSION}-{family}-{master_seed}-{index}', 'family': family, 'family_id': family_id, 'root_id': digest([family_id, world]), 'split': family_split(family_id), 'track': track, 'w …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`input_record` — L401**
```python
input_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`
- 반환 표현: `{'image': [image_path], 'instruction': '<image>\n' + instructions[view]}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`view_record` — L411**
```python
view_record(episode: dict[str, Any], image_path: str, view: str) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_record`, `json.dumps`, `enumerate`, `('qa', 'structure', 'grounded').index`
- 반환 표현: `{'id': episode['id'] + '-' + view, 'root_id': episode['root_id'], 'family_id': episode['family_id'], 'split': episode['split'], 'view': view, 'source_kind': episode['source_kind'], 'par …`

**`validate_episode` — L444**
```python
validate_episode(episode: dict[str, Any]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `semantic_family`, `ValueError`, `family_split`, `len`, `set`, `ord`, `fsm_identifiability`, `expected_quality.items`, `verify_fsm`, `any` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`generate_dataset` — L488**
```python
generate_dataset(output: Path, per_family: int=100, smoke_roots: int=120, seed: int=3407, bridge_every: int=5) -> dict[str, Any]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `output.exists`, `FileExistsError`, `output.mkdir`, `Path(__file__).read_bytes`, `Path`, `hashlib.sha256(source_bytes).hexdigest`, `hashlib.sha256`, `(output / 'generator_source.py').write_bytes`, `Path(__file__).with_name('clean_spec.json').read_bytes` 외 38개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`, `FileExistsError`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`verify_dataset` — L596**
```python
verify_dataset(output: Path) -> dict[str, Any]
```
- 기존 설명: Verify closed local artifacts and independently recompute every canonical result.
- 직접 호출: `json.loads`, `(output / 'manifest.json').read_text`, `hashlib.sha256(source.read_bytes()).hexdigest`, `hashlib.sha256`, `source.read_bytes`, `ValueError`, `hashlib.sha256((output / 'clean_spec.json').read_bytes()).hexdigest`, `(output / 'clean_spec.json').read_bytes`, `(output / 'canonical_private.jsonl').read_text().splitlines`, `(output / 'canonical_private.jsonl').read_text` 외 17개
- 반환 표현: `{'status': 'LOCAL_VERIFIED', 'episodes': len(episodes), 'images': checked_images, 'records': checked_records, 'independent_solvers': 'PASS', 'family_splits': 'PASS', 'input_whitelist':  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L655**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `verify_dataset`, `generate_dataset`
- 반환 표현: `None`

### D18.149 `fine_tuning/prepare_controls.py`

[현재 소스](../../fine_tuning/prepare_controls.py) · SHA256 `7860133269b827504707c7b07fbac78b6b8fa091bc58d738e717f6d87362f93e`

모듈의 기존 설명: Prepare the frozen NVW 0% control on Pod; no downloads, inference or training.

**`select_trace_control` — L14**
```python
select_trace_control(parent_rows, reference_rows, seed=3407) -> unannotated
```
- 기존 설명: Replace only the approved reference's NVW quota, independently of scores.
- 직접 호출: `any`, `row.get`, `ValueError`, `prepare_training.select_approved_subset`, `len`, `prepare_training.order`, `audit.check_splits`
- 반환 표현: `control`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare` — L33**
```python
prepare(parent, reference, output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(parent).resolve`, `Path`, `Path(reference).resolve`, `output.exists`, `output.is_symlink`, `any`, `output.resolve().is_relative_to`, `output.resolve`, `ValueError`, `prepare_training.closed_pilot` 외 42개
- 반환 표현: `receipt`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L136**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `print`, `json.dumps`, `prepare`, `vars`, `parser.parse_args`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.150 `fine_tuning/prepare_sources.py`

[현재 소스](../../fine_tuning/prepare_sources.py) · SHA256 `a127b1352438803fca1a71ab72894831af922d5018a1caf60a0c26630f8e473b`

모듈의 기존 설명: Remote-only exact benchmark exclusion inventory and TRACE audit downloads.

**`prepare_benchmarks` — L17**
```python
prepare_benchmarks(task_root) -> unannotated
```
- 기존 설명: Acquire EVERY released Parquet at pinned revisions for exclusion only.
- 직접 호출: `os.environ.get`, `ValueError`, `Path`, `target.exists`, `json.loads`, `(target / 'coverage.json').read_text`, `sha`, `target.mkdir`, `HfApi`, `api.dataset_info` 외 15개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare_trace` — L76**
```python
prepare_trace(task_root, inventory_path, shards=4) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `json.loads`, `Path(inventory_path).read_text`, `r['path'].startswith`, `r['path'].endswith`, `sorted`, `len`, `ValueError`, `download_exact`
- 반환 표현: `download_exact(metadata['repo_id'], metadata['revision'], files, root / 'hf-cache' / 'hub', root / 'setup' / 'trace-downloads.json')`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L89**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `prepare_benchmarks`, `print`, `json.dumps`, `len`, `parser.error`, `str`, `prepare_trace`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.151 `fine_tuning/prepare_training.py`

[현재 소스](../../fine_tuning/prepare_training.py) · SHA256 `a0cb1a90d0563c1b443f9e14e619d4e1385c6afaa072870d4c6b837e5f417e06`

모듈의 기존 설명: Private CPU-only preparation; no HF downloads, inference, or training.

**`policy` — L26**
```python
policy() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `POLICY_PATH.read_text`
- 반환 표현: `json.loads(POLICY_PATH.read_text())`

**`registry` — L30**
```python
registry(path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `Path(path).read_text`, `Path`, `value.get`, `isinstance`, `ValueError`
- 반환 표현: `value`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`order` — L38**
```python
order(rows, seed=3407) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `hashlib.sha256(f"{seed}:COMMON:{row['id']}".encode()).hexdigest`, `hashlib.sha256`, `f"{seed}:COMMON:{row['id']}".encode`
- 반환 표현: `sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:COMMON:{row['id']}".encode()).hexdigest())`

**`build_exclusion_index` — L42**
```python
build_exclusion_index(coverage_path, output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `Path(coverage_path).read_text`, `Path`, `sources.benchmark_index`
- 반환 표현: `sources.benchmark_index(paths, coverage, output)`

**`trace_paths` — L48**
```python
trace_paths(downloads_path, cache_root) -> unannotated
```
- 기존 설명: Resolve only receipt-listed existing cache files and verify all exact bytes.
- 직접 호출: `json.loads`, `Path(downloads_path).read_text`, `Path`, `receipt.get`, `isinstance`, `re.fullmatch`, `ValueError`, `Path(cache_root).resolve`, `set`, `len` 외 11개
- 반환 표현: `paths`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`joint_counts` — L78**
```python
joint_counts(rows) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Counter`
- 반환 표현: `{track: {view: counts[track, view] for view in ('qa', 'structure', 'grounded')} for track in ('NVW-K', 'NVW-I', 'NVW-S')}`

**`verified_canonical_roots` — L84**
```python
verified_canonical_roots(rows, path) -> unannotated
```
- 기존 설명: Root IDs are world fingerprints, not caller-controlled episode labels.
- 직접 호출: `audit.read_rows`, `isinstance`, `episode.get`, `nvw.digest`, `ValueError`, `any`, `row.get`
- 반환 표현: `canonical`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`processor_contract` — L101**
```python
processor_contract(receipt) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `receipt.get`
- 반환 표현: `{key: receipt.get(key) for key in ('processor_revision', 'processor_class', 'chat_template_sha256', 'sequence_budget', 'episode_pixels', 'producer_sha256')}`

**`validate_joint_quotas` — L106**
```python
validate_joint_quotas(matrix, view_quotas, track_quotas=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `ValueError`, `matrix.values`, `any`, `type`, `row.values`, `sum`, `values.values`, `matrix.items`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`select_nvw` — L119**
```python
select_nvw(nvw_root, quotas=None, seed=3407, joint_quotas=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `nvw.verify_dataset`, `policy`, `validate_joint_quotas`, `audit.read_rows`, `(nvw_root / name).exists`, `universe.extend`, `audit.check_splits`, `audit.sha`, `tracks.get` 외 15개
- 반환 표현: `(order(selected, seed), {'independent_verification': verification, 'available_views': {view: sum((row[view] for row in available_joint.values())) for view in quotas}, 'available_joint_v …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`copy_image` — L164**
```python
copy_image(source_root, destination_root, name) -> unannotated
```
- 기존 설명: Reuse regular assets by exact hash; exclusive-create every new destination.
- 직접 호출: `audit.image_path`, `Path(destination_root).resolve`, `Path`, `target.resolve().is_relative_to`, `target.resolve`, `ValueError`, `parent.is_symlink`, `target.parent.mkdir`, `audit.sha`, `target.exists` 외 4개
- 반환 표현: `expected`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`freeze_approved` — L190**
```python
freeze_approved(raw_path, image_root, source_registry, benchmark_index, model_path, output, expected_records, expected_roots=None, stage='pilot', selection=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `registry`, `audit.read_rows`, `audit.check_splits`, `audit.check_lineage`, `audit.check_target_identifiability`, `audit.preflight_records`, `audit.approve`, `len`, `joint_counts` 외 18개
- 반환 표현: `report`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare_smoke` — L246**
```python
prepare_smoke(nvw_root, source_registry, benchmark_index, model_path, output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `nvw.verify_dataset`, `audit.read_rows`, `policy`, `len`, `Counter`, `any`, `ValueError`, `output.mkdir`, `audit.write_rows` 외 3개
- 반환 표현: `freeze_approved(raw_path, nvw_root, source_registry, benchmark_index, model_path, output, expected['smoke_views'], expected['smoke_roots'], 'smoke', {'independent_verification': verific …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare_pilot` — L264**
```python
prepare_pilot(nvw_root, trace_downloads, trace_cache_root, source_registry, benchmark_index, model_path, output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `output.exists`, `FileExistsError`, `policy`, `select_nvw`, `trace_paths`, `sources.trace_records`, `len`, `audit.dump`, `ValueError` 외 8개
- 반환 표현: `freeze_approved(raw_path, output, source_registry, benchmark_index, model_path, output, expected['pilot_records'], stage='pilot', selection=selection)`
- 명시적 예외: `FileExistsError`, `ValueError`; 전파되는 callee 예외는 별도.

**`complete_pilot` — L301**
```python
complete_pilot(parent, parent_source_root, trace_downloads, trace_cache_root, source_registry, benchmark_index, model_path, output) -> unannotated
```
- 기존 설명: Reapprove parent + delta into approved/records.jsonl, with assets/manifest at output root.
- 직접 호출: `Path`, `policy`, `registry`, `output.exists`, `ValueError`, `audit.sha`, `(parent / 'common_manifest.json').exists`, `json.loads`, `(parent / 'preparation_receipt.json').read_text`, `(parent / 'approved/manifest.json').read_text` 외 62개
- 반환 표현: `receipt`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`approve_delta` — L493**
```python
approve_delta(rows, image_root, registered, benchmark_index, model_path, output) -> unannotated
```
- 기존 설명: The same real processor, rights and contamination gate for either replacement source.
- 직접 호출: `Path`, `output.mkdir`, `audit.write_rows`, `audit.check_lineage`, `audit.check_target_identifiability`, `audit.preflight_records`, `audit.approve`, `audit.read_rows`, `audit.sha`, `len`
- 반환 표현: `(additions, approved, processor, evidence)`

**`select_approved_subset` — L514**
```python
select_approved_subset(rows, count, seed=3407) -> unannotated
```
- 기존 설명: Keep approved bytes and fixed source/track/view ratios; never select by score.
- 직접 호출: `type`, `ValueError`, `len`, `any`, `row.get`, `order`, `list`, `matrix.items`, `by_view.items`, `selected.extend` 외 2개
- 반환 표현: `order(selected, seed)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`closed_pilot` — L545**
```python
closed_pilot(parent) -> unannotated
```
- 기존 설명: Verify the immutable full parent and its actual approved component chain.
- 직접 호출: `Path`, `audit.sha`, `approved_records`, `json.loads`, `(parent / 'completion_receipt.json').read_text`, `manifest.get`, `completed.get`, `closed_disjoint_components`, `len`, `completed.get('source_binding', {}).get` 외 15개
- 반환 표현: `closed_disjoint_components(parent, rows, manifest, completed, frozen)`; `(rows, manifest, completed, frozen, component_files)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`closed_disjoint_components` — L611**
```python
closed_disjoint_components(parent, rows, manifest, completed, frozen) -> unannotated
```
- 기존 설명: Retain reconstruction checks when a repaired full parent is later subsetted.
- 직접 호출: `manifest.get`, `completed.get`, `completed.get('source_sha256', {}).get`, `len`, `manifest.get('contamination_audit', {}).get`, `inputs.get`, `original_files.get`, `policy`, `joint_counts`, `Counter` 외 13개
- 반환 표현: `(rows, manifest, completed, frozen, components)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`smoke_exclusion` — L675**
```python
smoke_exclusion(rows, smoke_records) -> unannotated
```
- 기존 설명: Exclude roots only; never admit a record from the supplied smoke ledger.
- 직접 호출: `Path`, `audit.image_path`, `audit.sha`, `audit.read_rows`, `any`, `isinstance`, `row.get`, `row[key].strip`, `Path(name).is_absolute`, `ValueError` 외 7개
- 반환 표현: `(eligible, exclusion, smoke_path, smoke_rows)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`subset_pilot` — L708**
```python
subset_pilot(parent, output, count, reason, exclude_smoke_records=None) -> unannotated
```
- 기존 설명: New common subset of the approved 10K; keep the full parent and protocol unchanged.
- 직접 호출: `Path`, `output.exists`, `isinstance`, `reason.strip`, `ValueError`, `closed_pilot`, `smoke_exclusion`, `select_approved_subset`, `policy`, `output.mkdir` 외 25개
- 반환 표현: `scale`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`disjoint_full_pilot` — L769**
```python
disjoint_full_pilot(parent, nvw_root, smoke_records, smoke_manifest, source_registry, benchmark_index, model_path, output) -> unannotated
```
- 기존 설명: Replace smoke-overlapping NVW only; approve a NEW full 10K, never expand a reduced run.
- 직접 호출: `Path`, `output.exists`, `output.resolve().is_relative_to`, `output.resolve`, `parent.resolve`, `ValueError`, `closed_pilot`, `verified_canonical_roots`, `audit.image_path`, `audit.sha` 외 63개
- 반환 표현: `receipt`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L959**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_subparsers`, `sub.add_parser`, `index.add_argument`, `command.add_argument`, `complete.add_argument`, `subset.add_argument`, `disjoint.add_argument`, `vars`, `parser.parse_args` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.152 `fine_tuning/professional_transfer.py`

[현재 소스](../../fine_tuning/professional_transfer.py) · SHA256 `46fe8fdcdd979e0dbfae8249b9f32cf8dda627287d9539bc21e12e0069d0f39d`

모듈의 기존 설명: Private ChartQA-human source-holdout preparation; HF payloads stay on RunPod.

**`training_index` — L30**
```python
training_index(specs) -> unannotated
```
- 기존 설명: Validate actual trained datasets, including the original frozen sources.
- 직접 호출: `isinstance`, `ValueError`, `set`, `Path`, `json.loads`, `manifest_path.read_text`, `metadata_path.read_text`, `audit.sha`, `manifest.get`, `manifest.get('contamination_audit', {}).get` 외 29개
- 반환 표현: `{'pixels': pixels, 'dhashes': dhashes, 'texts': texts, 'ancestors': ancestors, 'evidence': evidence}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`observation` — L105**
```python
observation(row, source_index) -> unannotated
```
- 기존 설명: Gold-blind projection. The caller must not use labels to replace a row.
- 직접 호출: `type`, `ValueError`, `isinstance`, `query.strip`, `entry.get`, `Image.open`, `io.BytesIO`, `audit.image_fingerprint`, `digest`, `hashlib.sha256(raw).hexdigest` 외 1개
- 반환 표현: `None`; `{'id': 'chartqa-human-' + identity, 'query': query, 'source_index': source_index, 'family_id': 'chartqa-chart-' + fingerprint['pixels_sha256'], 'source_bytes_sha256': hashlib.sha256(raw …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`select` — L126**
```python
select(observations, index, count=60, seed=3407) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `type`, `ValueError`, `len`, `sorted`, `digest`, `audit.text_hash`, `any`, `(int(row['dhash'], 16) ^ int(value, 16)).bit_count`, `int`, `rejected.append` 외 2개
- 반환 표현: `(chosen, rejected)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`freeze` — L148**
```python
freeze(rows, index, output, source_receipt, count=60, seed=3407) -> unannotated
```
- 기존 설명: Pure offline builder also exercised with independently authored fixtures.
- 직접 호출: `enumerate`, `observation`, `select`, `isinstance`, `any`, `value.strip`, `ValueError`, `Path`, `out.mkdir`, `(out / 'images').mkdir` 외 14개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare` — L197**
```python
prepare(task_root, training_evidence, output, count=60, seed=3407) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `os.environ.get`, `ValueError`, `Path(output).exists`, `Path`, `json.loads`, `LOCK.read_text`, `training_index`, `Path(training_evidence).read_text`, `download_exact`, `path.stat` 외 5개
- 반환 표현: `freeze(rows, index, output, {'lock_sha256': audit.sha(LOCK), 'download_receipt_sha256': audit.sha(receipt_path), **lock}, count, seed)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L218**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `prepare`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.153 `fine_tuning/recover_evaluation.py`

[현재 소스](../../fine_tuning/recover_evaluation.py) · SHA256 `14c914baebdbf5bf8a8d4d991f8524004a4ef4b5c5a6aec371bb12083a46f8ba`

모듈의 기존 설명: Recover a terminated F0 summary-save failure, never repair or regenerate its raw rows.

**`process_gone` — L17**
```python
process_gone(launch) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `int`, `Path`, `stat.exists`, `stat.read_text().rsplit(')', 1)[1].split`, `stat.read_text().rsplit`, `stat.read_text`, `str`
- 반환 표현: `True`; `stat.read_text().rsplit(')', 1)[1].split()[19] != str(launch['pid_birth'])`

**`recover` — L26**
```python
recover(parent, cohort, protocol_path, launch_path, model_receipt_path, log_path, output, owned_pod_id) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `json.loads`, `Path(launch_path).read_text`, `Path(model_receipt_path).read_text`, `Path(protocol_path).read_text`, `(cohort / 'manifest.json').read_text`, `os.environ.get`, `process_gone`, `command.index`, `argument` 외 29개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L120**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `print`, `json.dumps`, `recover`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.154 `fine_tuning/report.py`

[현재 소스](../../fine_tuning/report.py) · SHA256 `64667fa51d1b41128d1f574921f77c6e307b398fb77eb0e6e7b379bdeaf43d04`

모듈의 기존 설명: Private saved-result report; no model calls, training, or public baseline replacement.

**`load_evaluation` — L26**
```python
load_evaluation(path: Path, baseline: bool=False) -> tuple[dict, list[dict]]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(path / 'summary.json').read_text`, `read_rows`, `row.get`, `len`, `bool`, `sum`, `summary.get`, `sha`, `any` 외 2개
- 반환 표현: `(summary, rows)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`compare_inputs` — L49**
```python
compare_inputs(reference: list[dict], rows: list[dict]) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `zip`, `any`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_counts` — L60**
```python
paired_counts(reference: list[dict], rows: list[dict]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Counter`, `zip`, `type`, `before.get`, `after.get`, `ValueError`, `dict`, `len`, `sum`
- 반환 표현: `dict(counts, count=len(rows), net=counts['gain'] - counts['loss'], f0_correct=sum((row['correct'] for row in reference)), ft_correct=sum((row['correct'] for row in rows)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`external_groups` — L73**
```python
external_groups(reference: list[dict], rows: list[dict]) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `sorted`, `any`, `before.get`, `after.get`, `zip`, `ValueError`, `enumerate`, `paired_counts`, `type`, `math.isfinite` 외 2개
- 반환 표현: `groups`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`frozen_pilot_scale` — L106**
```python
frozen_pilot_scale(controller: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `path.exists`, `json.loads`, `path.read_text`, `frozen.get`, `scale.get`, `type`, `ValueError`, `declared.get`, `dict`, `sha`
- 반환 표현: `{'original_records': 10000, 'actual_records': 10000, 'scaled': False, 'source_counts': {}, 'legacy_default': True}`; `dict(scale, freeze_sha256=sha(path), protocol_sha256=frozen.get('protocol_sha256'), records_sha256=frozen.get('pilot_records_sha256'), manifest_sha256=frozen.get('pilot_manifest_sha256' …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`checkpoint_evidence` — L131**
```python
checkpoint_evidence(controller: Path, arm: str, journal: list[dict], scale: dict | None=None) -> tuple[dict, dict | None]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `metadata_path.exists`, `json.loads`, `metadata_path.read_text`, `metadata.get`, `ValueError`, `files.items`, `(checkpoint / name).resolve`, `path.is_relative_to`, `checkpoint.resolve`, `sha` 외 6개
- 반환 표현: `({'status': '未完成', 'reason': 'Pilot checkpoint missing'}, None)`; `({'status': 'COMPLETED' if complete else '未完成', 'path': str(checkpoint.resolve()), 'metadata_sha256': sha(metadata_path), 'checkpoint_files': files, 'exposure': metadata.get('exposure') …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verified_reuse` — L160**
```python
verified_reuse(controller: Path, journal: list[dict], baselines: dict) -> tuple[dict, dict, dict]
```
- 기존 설명: Recheck original absolute artifacts; receipts never stand in for parent evidence.
- 직접 호출: `training_path.exists`, `evaluation_path.exists`, `ValueError`, `json.loads`, `(controller / 'pilot_input_freeze.json').read_text`, `frozen_path`, `str`, `freeze.get`, `Path(freeze['image_root']).is_absolute`, `Path` 외 15개
- 반환 표현: `({}, {}, {})`; `(training, evaluations, evidence)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`build_report` — L224**
```python
build_report(controller: Path, baseline_mmmu: Path, baseline_external: Path, output: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `load_evaluation`, `Counter`, `ValueError`, `(controller / 'journal.jsonl').exists`, `read_rows`, `frozen_pilot_scale`, `verified_reuse`, `scale.get`, `list`, `limits.append` 외 44개
- 반환 표현: `result`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L347**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `build_report`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.155 `fine_tuning/run_bridge_experiment.py`

[현재 소스](../../fine_tuning/run_bridge_experiment.py) · SHA256 `e1c97f99c42a2a51896410bdd97e5a39f3aeccd3806cd4e871bdbd55da7a62da`

모듈의 기존 설명: Fixed-module bounded Graph/CS or DC-circuit A0-A3 probes, not full G7.

**`circuit_metric_gate` — L24**
```python
circuit_metric_gate(cohort, manifest) -> unannotated
```
- 기존 설명: A Circuit label cannot route rebound gold through the permissive V8 QA path.
- 직접 호출: `manifest.get`, `audit.read_rows`, `audit.sha`, `len`, `any`, `entry.get`, `reference.get`, `zip`, `ValueError`
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`result_gate` — L41**
```python
result_gate(output, cohort, protocol, checkpoint=None, evaluator_source=None, parent_evaluator_source=None) -> unannotated
```
- 기존 설명: Validate identity, observed inputs and independently replay stored CPU scores.
- 직접 호출: `runner.evaluation_gate`, `runner.cohort_gate`, `circuit_metric_gate`, `audit.read_rows`, `manifest.get`, `Path`, `summary.get`, `audit.sha`, `Path(summary.get('checkpoint', '')).resolve`, `checkpoint.resolve` 외 16개
- 반환 표현: `(summary, rows)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`graph_groups` — L104**
```python
graph_groups(cohort, before, after) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `audit.read_rows`, `ValueError`, `sorted`, `enumerate`, `external_groups`
- 반환 표현: `grouped`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`training_bundle` — L117**
```python
training_bundle(args) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `bridge.inputs`, `frozen.update`, `any`, `datasets.values`, `ValueError`, `runner.cohort_gate`, `circuit_metric_gate`, `graph.get`, `type`, `audit.sha` 외 12개
- 반환 표현: `(datasets, frozen)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`reuse_a3` — L165**
```python
reuse_a3(parent, datasets, args, reference, cfg, merge_size) -> unannotated
```
- 기존 설명: Rebind locally recovered paths without pretending their remote paths still exist.
- 직접 호출: `ValueError`, `runner.closed_controller`, `verify_arm_training`, `bridge.verify_processor_exposure`, `json.loads`, `(parent / 'source_freeze.json').read_text`, `(parent / 'input_freeze.json').read_text`, `any`, `sources.get`, `audit.sha` 외 21개
- 반환 표현: `(output, metadata, frozen)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`run` — L211**
```python
run(args, controller) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `process_gone`, `ValueError`, `runner.closed_controller`, `audit.sha`, `Path(__file__).with_name`, `Path`, `training_bundle`, `frozen.update`, `json.loads`, `reference_path.read_text` 외 53개
- 반환 표현: `{'status': 'BUDGET_STOP', 'GPU_started': False, 'reason': 'First evaluation and recovery reserve cannot fit'}`; `{'status': f'BOUNDED_{prefix}_FIXED_MODULE_A0_A3_EVALUATED', 'arm': args.arm, 'training_exposures': {c: m['exposure'] for c, m in metadata.items()}, 'results': results, 'contrasts': con …`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

**`main` — L375**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `os.environ.get`, `args.previous_birth.isdecimal`, `args.model_path.is_dir`, `parser.error`, `runner.effective_deadline`, `runner.utc`, `runner.Controller` 외 12개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `SystemExit`; 전파되는 callee 예외는 별도.

### D18.156 `fine_tuning/run_bridge_smoke.py`

[현재 소스](../../fine_tuning/run_bridge_smoke.py) · SHA256 `867dd691e9eaab489941b674aa80ab8d3712786be00acf21bae0455028f51417`

모듈의 기존 설명: Bounded fresh-F0 FL GPU input/update/reload check, not fullG7 or a score experiment.

**`inputs` — L21**
```python
inputs(approval: Path, image_root: Path, child: Path, condition: str='A3') -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `approved_records`, `audit.training_input_contract`, `validate_training_input_contract`, `json.loads`, `(image_root / 'manifest.json').read_text`, `pool.get`, `circuit_bridge.verify`, `audit.read_rows`, `candidates.items` 외 22개
- 반환 표현: `(rows, manifest_path, records, contract, frozen)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_processor_exposure` — L73**
```python
verify_processor_exposure(output: Path, approval: Path, condition: str, metadata: dict, merge_size: int) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(approval / f'{condition}_processor.json').read_text`, `audit.read_rows`, `any`, `sum`, `math.prod`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`run` — L83**
```python
run(args, controller) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `process_gone`, `ValueError`, `runner.closed_controller`, `inputs`, `rows[0].get`, `json.loads`, `reference_path.read_text`, `reference.get`, `runner.exposure_matches`, `frozen.update` 외 25개
- 반환 표현: `{'status': 'BUDGET_STOP', 'GPU_training_started': False, 'reason': '900sec bounded train/reload plus30sec termination and120sec recovery margin do not fit'}`; `{'status': 'BOUNDED_GRAPH_A3_GPU_SMOKE_COMPLETED', 'GPU_training_started': True, 'all84_inputs_forward_backward_updated': True, 'fresh_process_reload_verified': True, 'exposure': metada …`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

**`main` — L151**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `os.environ.get`, `args.previous_birth.isdecimal`, `args.model_path.is_dir`, `parser.error`, `runner.effective_deadline`, `runner.utc`, `runner.Controller` 외 12개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `SystemExit`; 전파되는 callee 예외는 별도.

### D18.157 `fine_tuning/run_control.py`

[현재 소스](../../fine_tuning/run_control.py) · SHA256 `039dd9c1e3bccb1e130e3274ffd473b4b2083de3dd64b7e698ff45acfabbd616`

모듈의 기존 설명: Additional FL-only NVW 0/10 mixture pilot; never the original 10K or G7.

**`control_inputs` — L19**
```python
control_inputs(parent: Path, reference: Path, control: Path, smoke_root: Path, smoke_image_root: Path) -> unannotated
```
- 기존 설명: Reconstruct the approved substitution; copied receipts alone are insufficient.
- 직접 호출: `prepare_training.closed_pilot`, `approved_records`, `json.loads`, `receipt_path.read_text`, `(reference / 'subset_receipt.json').read_text`, `audit.sha`, `audit.image_path`, `len`, `receipt.get`, `Path(receipt.get('reference', '')).resolve` 외 25개
- 반환 표현: `(control_rows, reference_rows, smoke_rows, frozen)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_training` — L106**
```python
verify_training(output: Path, rows: list, manifest: Path, cfg: dict, reference: dict) -> unannotated
```
- 기존 설명: Verify the fresh FL journals and checkpoint without inventing a closed four-arm run.
- 직접 호출: `len`, `ValueError`, `verify_fl_training`
- 반환 표현: `verify_fl_training(output, rows, manifest.parent / 'approved/records.jsonl', manifest, cfg, reference)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_fl_training` — L113**
```python
verify_fl_training(output: Path, rows: list, records: Path, manifest: Path, cfg: dict, reference: dict, input_contract: dict | None=None) -> unannotated
```
- 기존 설명: Shared complete-epoch FL receipt check; the control wrapper keeps its1600 gate.
- 직접 호출: `verify_arm_training`
- 반환 표현: `verify_arm_training(output, rows, records, manifest, cfg, reference, 'FL', input_contract)`

**`checkpoint_layout` — L119**
```python
checkpoint_layout(checkpoint: Path, metadata: dict) -> unannotated
```
- 기존 설명: Reject unlisted files and automatic-load paths belonging to another arm.
- 직접 호출: `set`, `str`, `p.relative_to`, `checkpoint.rglob`, `p.is_file`, `any`, `p.is_symlink`, `(checkpoint / 'adapter').exists`, `(checkpoint / 'mergers.pt').exists`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_arm_training` — L130**
```python
verify_arm_training(output: Path, rows: list, records: Path, manifest: Path, cfg: dict, reference: dict, arm: str, input_contract: dict | None=None) -> unannotated
```
- 기존 설명: One complete-epoch journal/payload check for the actual selective partitions.
- 직접 호출: `ValueError`, `set`, `json.loads`, `(checkpoint / 'metadata.json').read_text`, `list`, `range`, `len`, `random.Random(3407).shuffle`, `random.Random`, `audit.read_rows` 외 25개
- 반환 표현: `metadata`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`require_idle_gpu` — L197**
```python
require_idle_gpu() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `subprocess.run`, `probe.stdout.strip`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`run` — L204**
```python
run(args, controller) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `runner.closed_controller`, `json.loads`, `args.protocol.read_text`, `Path(__file__).with_name('protocol.json').read_text`, `Path(__file__).with_name`, `Path`, `ValueError`, `control_inputs`, `runner.pilot_scale_evidence`, `(args.reference / 'common_manifest.json').read_text` 외 49개
- 반환 표현: `{'status': 'BUDGET_STOP', 'GPU_training_started': False, 'required_seconds_estimate': estimate}`; `{'status': report['status'], 'GPU_training_started': True, 'report_sha256': audit.sha(controller.output / 'report.json')}`
- 명시적 예외: `ValueError`, `TimeoutError`; 전파되는 callee 예외는 별도.

**`main` — L320**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `os.environ.get`, `args.model_path.is_dir`, `parser.error`, `runner.effective_deadline`, `runner.utc`, `runner.Controller`, `Path` 외 11개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `SystemExit`; 전파되는 callee 예외는 별도.

### D18.158 `fine_tuning/run_experiment.py`

[현재 소스](../../fine_tuning/run_experiment.py) · SHA256 `e830c2e3ab5ce41b67c0fc6da1351b225cfbf1fd953754b347090ba50183051d`

모듈의 기존 설명: Owned-Pod subprocess controller; frozen sources/data and explicit GPU/balance cutoff.

**`source_snapshot` — L27**
```python
source_snapshot(root: Path) -> dict[str, str]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `str`, `p.relative_to`, `sha`, `sorted`, `root.rglob`, `set`
- 반환 표현: `{str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*.py')) if not {'runs', 'generated', '__pycache__'} & set(p.relative_to(root).parts)}`

**`assert_source_frozen` — L32**
```python
assert_source_frozen(root: Path, frozen: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `source_snapshot`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`exposure_matches` — L37**
```python
exposure_matches(metadata: dict) -> bool
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `metadata.get`, `metadata.get('exposure', {}).get`
- 반환 표현: `metadata.get('epoch_exposure_complete') is True and metadata.get('exposure', {}).get('records') == metadata.get('planned_records') and (metadata.get('consumed_order_sha256') == metadata …`

**`assert_same_exposure` — L44**
```python
assert_same_exposure(arms: dict) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `arms.values`, `any`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`pilot_estimate` — L50**
```python
pilot_estimate(smokes: dict, pilot_records: int, evaluation_seconds: float, margin: float=2.0, reused_training=(), reused_evaluations=(), evaluation_by_cohort=None) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.isfinite`, `set`, `ValueError`, `any`, `exposure_matches`, `sum`, `len`
- 반환 표현: `margin * total + evaluation + 150 * (4 - len(set(reused_training)))`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`pilot_scale_evidence` — L73**
```python
pilot_scale_evidence(rows: list[dict], manifest: dict, original_records: int, explicit_records: int | None, smoke_records: int, seed: int) -> dict
```
- 기존 설명: An explicit approved common subset changes training exposure, never inference.
- 직접 호출: `len`, `ValueError`, `type`, `dict`, `manifest.get`, `scale.get`, `any`, `isinstance`, `scale[key].strip`, `re.fullmatch` 외 3개
- 반환 표현: `dict(evidence, source_counts=manifest.get('source_counts', {}), nvw_tracks=manifest.get('nvw_tracks', {}), nvw_joint_views=manifest.get('nvw_joint_views', {}))`; `dict(evidence, pilot_scale=scale, source_counts=sources, nvw_tracks=tracks, nvw_joint_views=views, nvw_view_counts=view_counts, original_10K_execution_completed=False)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`closed_controller` — L115**
```python
closed_controller(parent: Path, evaluation_only: bool=False) -> tuple[dict, list, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `status_path.read_text`, `read_rows`, `terminal.add`, `status.get`, `journal[-1].get`, `ValueError`, `str`, `path.resolve`, `sha`
- 반환 표현: `(status, journal, frozen)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`reuse_training` — L130**
```python
reuse_training(parent: Path, rows: list[dict], records: Path, manifest: Path, cfg: dict, seed: int, phase: str) -> tuple[dict, dict, dict]
```
- 기존 설명: Verify closed train + fresh reload; partial/unclosed arms never supply weights.
- 직접 호출: `closed_controller`, `sha`, `Path(__file__).with_name`, `Path`, `json.loads`, `(parent / 'source_freeze.json').read_text`, `any`, `source_freeze.get`, `current_sources.items`, `ValueError` 외 41개
- 반환 표현: `(smokes, frozen, receipt)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`reuse_smoke` — L238**
```python
reuse_smoke(parent: Path, rows: list[dict], records: Path, manifest: Path, cfg: dict, seed: int) -> tuple[dict, dict, dict]
```
- 기존 설명: Reuse G4 evidence, never initialize a pilot with smoke-trained weights.
- 직접 호출: `len`, `ValueError`, `reuse_training`
- 반환 표현: `reuse_training(parent, rows, records, manifest, cfg, seed, 'smoke')`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`reuse_pilot` — L246**
```python
reuse_pilot(parent: Path, rows: list[dict], records: Path, manifest: Path, cfg: dict, seed: int, protocol: Path, image_root: Path) -> tuple[dict, dict, dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `freeze_path.read_text`, `freeze.get`, `sha`, `Path(freeze.get('image_root', '')).resolve`, `Path`, `image_root.resolve`, `ValueError`, `reuse_training`, `str` 외 2개
- 반환 표현: `(pilots, frozen, receipt)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `Controller` — L262**

기저: `object`.

**`Controller.__init__` — L263**
```python
Controller.__init__(self, output: Path, deadline: dt.datetime, source_root: Path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `output.mkdir`, `source_snapshot`, `dump`, `deadline.isoformat`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`Controller.remaining` — L270**
```python
Controller.remaining(self) -> float
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `(self.deadline - dt.datetime.now(dt.timezone.utc)).total_seconds`, `dt.datetime.now`
- 반환 표현: `(self.deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()`

**`Controller.event` — L273**
```python
Controller.event(self, **value) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `dt.datetime.now(dt.timezone.utc).isoformat`, `dt.datetime.now`, `(self.output / 'journal.jsonl').open`, `stream.write`, `json.dumps`, `stream.flush`, `os.fsync`, `stream.fileno`, `print`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`Controller.stage` — L281**
```python
Controller.stage(self, name: str, command: list[str], maximum_seconds: float | None=None, termination_grace_seconds: float=15) -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `math.isfinite`, `ValueError`, `assert_source_frozen`, `self.remaining`, `TimeoutError`, `min`, `self.event`, `time.monotonic`, `(self.output / (name + '.log')).open`, `signal.getsignal` 외 10개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`, `TimeoutError`, `re-raise`, `RuntimeError`; 전파되는 callee 예외는 별도.

**`cohort_gate` — L339**
```python
cohort_gate(cohort: Path, expected_count: int | None=None) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(cohort / 'manifest.json').read_text`, `manifest.get`, `ValueError`, `sha`
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`evaluation_gate` — L351**
```python
evaluation_gate(output: Path, cohort: Path, protocol: Path, baseline: bool=False) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(output / 'summary.json').read_text`, `(cohort / 'manifest.json').read_text`, `summary.get`, `sha`, `ValueError`, `(cohort / 'inputs.jsonl').read_text().splitlines`, `(cohort / 'inputs.jsonl').read_text`, `(output / 'predictions.jsonl').read_text().splitlines`, `(output / 'predictions.jsonl').read_text` 외 4개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`paired_input_gate` — L371**
```python
paired_input_gate(output: Path, baseline: Path, cohort: Path, protocol: Path) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `evaluation_gate`, `read_rows`, `zip`, `any`, `ValueError`, `json.dumps`, `observed.append`, `len`, `list`, `str` 외 5개
- 반환 표현: `{'status': 'PASS', 'kind': 'PAIRED_ACTUAL_INPUT_EQUIVALENCE', 'count': len(observed), 'fields': list(PAIRED_INPUT_FIELDS), 'baseline_result': str(baseline.resolve()), 'baseline_summary_ …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`reuse_pilot_evaluations` — L396**
```python
reuse_pilot_evaluations(parent: Path, checkpoints: dict, baselines: dict, cohorts: dict, protocol: Path) -> tuple[dict, dict, dict]
```
- 기존 설명: Reference closed exact-checkpoint results; never modify or copy old outputs.
- 직접 호출: `closed_controller`, `json.loads`, `(parent / 'source_freeze.json').read_text`, `cohorts.items`, `event.get`, `output.exists`, `ValueError`, `events[-1].get`, `cohort_gate`, `evaluation_gate` 외 16개
- 반환 표현: `(results, frozen, {'status': 'PASS', 'kind': 'CLOSED_PILOT_EVALUATION_REUSE', 'parent': str(parent.resolve()), 'parent_status': status['status'], 'files': frozen, 'arms': evidence})`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`run` — L440**
```python
run(args, controller: Controller) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `args.protocol.read_text`, `approved_records`, `len`, `ValueError`, `getattr`, `type`, `max`, `cohort_gate`, `subprocess.run` 외 22개
- 반환 표현: `{'status': 'BLOCKED', 'smoke_completed': True, 'pilot_started': False, 'reason': f'Common pilot approval missing or invalid: {exc}'}`; `{'status': 'SMOKE_COMPLETE_PILOT_BUDGET_STOP', 'pilot_started': False, 'required_seconds_estimate': estimate, 'remaining_seconds': controller.remaining()}`; `{'status': 'FOUR_ARM_PILOT_EVALUATED', 'pilot_started': True, 'pilot_records': len(pilot_rows), 'original_pilot_records': cfg['pilot_records'], 'original_10K_execution_completed': len(p …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L619**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `math.isfinite`, `parser.error`, `os.environ.get`, `effective_deadline`, `utc`, `Controller`, `Path` 외 12개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `SystemExit`; 전파되는 callee 예외는 별도.

### D18.159 `fine_tuning/runpod_g6.py`

[현재 소스](../../fine_tuning/runpod_g6.py) · SHA256 `5fe10bee0b617cd518da54b407ed88fb312ec543d5b87c531ede3740fea2825e`

모듈의 기존 설명: Fresh full-10K G6 from a pinned MMDL_test clone, with bounded preparation too.

**`checkout` — L17**
```python
checkout(repo, commit) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `re.fullmatch`, `ValueError`, `git`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`expected_shortfall` — L30**
```python
expected_shortfall(raw) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(raw / 'preparation_receipt.json').read_text`, `receipt.get`, `type`, `(raw / 'common_manifest.json').exists`, `ValueError`
- 반환 표현: `receipt`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`run` — L41**
```python
run(args, controller, repo) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `(root / 'setup/model.json').read_text`, `setup.get`, `ValueError`, `(repo / 'manifests/models/baseline.json').read_text`, `setup.get('manifest', {}).get`, `Path`, `model.resolve().is_relative_to`, `model.resolve`, `sha` 외 8개
- 반환 표현: `result`; `{'status': 'FOUR_ARM_PILOT_EVALUATED', 'report': str(root / 'reports/g6')}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L124**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `Path(__file__).resolve`, `Path`, `checkout`, `root.is_absolute`, `root.resolve`, `root.is_relative_to`, `parser.error` 외 16개
- 반환 표현: `None`
- 명시적 예외: `SystemExit`; 전파되는 callee 예외는 별도.

### D18.160 `fine_tuning/sources.py`

[현재 소스](../../fine_tuning/sources.py) · SHA256 `effcab0b9c7483ed52119df6b37f2c98dfd641c5f7c41301c923bd7ce029b95b`

모듈의 기존 설명: Metadata locally; exact approved external downloads only on an owned RunPod.

**`inspect` — L32**
```python
inspect(output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `HfApi`, `SOURCES.items`, `api.dataset_info`, `info.card_data.to_dict`, `type`, `dump`
- 반환 표현: `rows`

**`download_exact` — L53**
```python
download_exact(repo, revision, files, cache, destination) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `os.environ.get`, `ValueError`, `HfApi().dataset_info`, `HfApi`, `len`, `Path`, `rel.is_absolute`, `hf_hub_download`, `str`, `sha` 외 4개
- 반환 표현: `paths`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`benchmark_index` — L80**
```python
benchmark_index(paths, coverage, output) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `set`, `isinstance`, `coverage.get`, `ValueError`, `str`, `Path(entry['path']).resolve`, `Path`, `entry.get`, `re.fullmatch`, `Path(path).resolve` 외 32개
- 반환 표현: `{k: value[k] for k in ('status', 'coverage', 'record_count')}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`typed_trace_answer` — L170**
```python
typed_trace_answer(value) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `isinstance`, `json.loads`, `set`, `ValueError`, `math.isfinite`, `bool`, `result.strip`, `re.fullmatch`
- 반환 표현: `answer`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`trace_family_groups` — L188**
```python
trace_family_groups(rows, seed) -> unannotated
```
- 기존 설명: Union actual task/scene/grammar identifiers before any split/augmentation.
- 직접 호출: `find`, `list`, `components.setdefault(find(token), []).append`, `components.setdefault`, `sorted`, `components.get`, `hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest`, `hashlib.sha256`, `json.dumps(tokens, separators=(',', ':')).encode`, `json.dumps` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`trace_records` — L212**
```python
trace_records(paths, output, seed=3407, train_limit=9000) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path`, `root.mkdir`, `images.mkdir`, `set`, `sha`, `pq.ParquetFile(path).iter_batches`, `pq.ParquetFile`, `batch.to_pylist`, `item.get`, `isinstance` 외 42개
- 반환 표현: `train + remaining`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L304**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_subparsers`, `sub.add_parser`, `metadata.add_argument`, `parser.parse_args`, `inspect`, `print`, `json.dumps`, `v.get`, `result.items`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.161 `fine_tuning/tabmwp.py`

[현재 소스](../../fine_tuning/tabmwp.py) · SHA256 `4ea8a28f96744878715f7abe3aefb66c3e41692f986392143814cb01436d1789`

모듈의 기존 설명: Teacher-free TabMWP preparation; never grants training approval or fetches images.

**`digest` — L32**
```python
digest(value) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':') …`, `hashlib.sha256`, `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`

**`_verify_blob` — L37**
```python
_verify_blob(data, name) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha1(f'blob {len(data)}\x00'.encode() + data).hexdigest`, `hashlib.sha1`, `f'blob {len(data)}\x00'.encode`, `len`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`acquire` — L44**
```python
acquire(source_root) -> unannotated
```
- 기존 설명: Download only six small, pinned source documents into a new private root.
- 직접 호출: `Path`, `root.exists`, `FileExistsError`, `ARTIFACTS.items`, `urllib.request.urlopen`, `response.read`, `_verify_blob`, `root.mkdir`, `payloads.items`, `path.parent.mkdir` 외 5개
- 반환 표현: `receipt`
- 명시적 예외: `FileExistsError`; 전파되는 callee 예외는 별도.

**`load_source` — L74**
```python
load_source(source_root) -> unannotated
```
- 기존 설명: Require the immutable official blob identities, not a self-asserted receipt.
- 직접 호출: `Path`, `json.loads`, `(root / 'source_receipt.json').read_text`, `receipt.get`, `set`, `ValueError`, `path.is_symlink`, `path.resolve().is_relative_to`, `path.resolve`, `root.resolve` 외 14개
- 반환 표현: `(problems, receipt)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`table_facts` — L105**
```python
table_facts(problem) -> unannotated
```
- 기존 설명: Preserve column/cell order; require agreement of both native table forms.
- 직접 호출: `problem.get`, `isinstance`, `any`, `columns.values`, `ValueError`, `list`, `len`, `zip`, `type`, `cell.strip` 외 2개
- 반환 표현: `{'title': title, 'headers': visible_headers, 'rows': rows}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`family_id` — L132**
```python
family_id(facts) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `cell.strip`, `digest`
- 반환 표현: `'tabmwp-table-' + digest(ordered)`

**`paired_records` — L139**
```python
paired_records(pid, problem, facts, native_split) -> unannotated
```
- 기존 설명: Same source image/question/ID; only assistant supervision changes.
- 직접 호출: `problem.get`, `isinstance`, `pid.isdecimal`, `question.strip`, `answer.strip`, `ValueError`, `len`, `any`, `c.strip`, `set` 외 9개
- 반환 표현: `{variant: model_input_record(dict(common, target_text=json.dumps(target, ensure_ascii=False, separators=(',', ':')))) for variant, target in targets.items()}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`plan` — L178**
```python
plan(problems, excluded_train_families=()) -> unannotated
```
- 기존 설명: Scan all original splits before selecting, excluding cross-split families.
- 직접 호출: `set`, `ValueError`, `defaultdict`, `problems[split].items`, `seen.add`, `table_facts`, `quarantine.append`, `str`, `families[family_id(facts)].add`, `family_id` 외 12개
- 반환 표현: `(variants, {'seed': SEED, 'selected_ids': selected_ids, 'candidate_counts': candidates, 'native_counts': {k: len(v) for k, v in problems.items()}, 'cross_split_family_count': len(leakin …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`exclusion_policy` — L236**
```python
exclusion_policy(source_root, problems, receipt_path) -> unannotated
```
- 기존 설명: Validate a historical candidate scan, never convert it into an audit approval.
- 직접 호출: `Path(source_root).resolve`, `Path`, `json.loads`, `receipt_path.read_text`, `scan.get`, `any`, `ValueError`, `plan`, `isinstance`, `len` 외 35개
- 반환 표현: `{'policy': 'PRETRAINING_CONSERVATIVE_CANDIDATE_FAMILY_EXCLUSION_V1', 'reason': 'Exclude candidate table families before a new fixed-quota preparation; not duplicate proof or approval',  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare` — L336**
```python
prepare(source_root, output_root, exclusion_receipt=None) -> unannotated
```
- 기존 설명: Freeze paired proposals, not approved training inputs; images remain pending.
- 직접 호출: `Path`, `output.exists`, `FileExistsError`, `load_source`, `exclusion_policy`, `plan`, `ValueError`, `output.mkdir`, `(output / 'exclusion_receipt.json').open`, `stream.write` 외 9개
- 반환 표현: `manifest`
- 명시적 예외: `FileExistsError`, `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L386**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `parser.error`, `acquire`, `prepare`, `print`, `json.dumps`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.162 `fine_tuning/tabmwp_evaluate.py`

[현재 소스](../../fine_tuning/tabmwp_evaluate.py) · SHA256 `4d2d9eb9803205e5743da8736d8d5fc051e87c6a8c79442005ecc80321b48693`

모듈의 기존 설명: Frozen teacher-free TabMWP A/B external evaluation, separate from MMMU/V8 runs.

**`_indexed` — L46**
```python
_indexed(values, key='sample_id') -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `recovery.unique`
- 반환 표현: `recovery.unique(values, key)`

**`_new_destination` — L50**
```python
_new_destination(path, sources=()) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(path).absolute`, `Path`, `path.exists`, `path.is_symlink`, `FileExistsError`, `path.resolve`, `resolved.is_relative_to`, `ValueError`, `any`, `Path(source).resolve` 외 1개
- 반환 표현: `path`
- 명시적 예외: `FileExistsError`, `ValueError`; 전파되는 callee 예외는 별도.

**`_input_key` — L63**
```python
_input_key(text, hashes) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `digest`
- 반환 표현: `digest({'user_text': text, 'ordered_image_sha256': hashes})`

**`_training_inputs` — L67**
```python
_training_inputs(records_path, image_root, native_ids) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `read_rows`, `_indexed`, `row.get`, `len`, `set`, `any`, `ValueError`, `str`, `sha256_file`, `families.add` 외 9개
- 반환 표현: `(families, roots, inputs, sources, evidence)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`prepare` — L95**
```python
prepare(prepared_root, images_root, retention_root, training_records, training_image_root, output, gold_output) -> unannotated
```
- 기존 설명: Freeze fixed development requests, with sealed data used only for exclusion/validation.
- 직접 호출: `map`, `_new_destination`, `tabmwp_images.selection`, `sorted`, `images_root.glob`, `read_json`, `tabmwp_images.verify_tree`, `receipt.get`, `digest`, `set` 외 39개
- 반환 표현: `manifest`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`load_cohort` — L245**
```python
load_cohort(data_root) -> unannotated
```
- 기존 설명: Gold is intentionally not an argument, opened file, or inference sample field.
- 직접 호출: `Path`, `read_json`, `manifest.get`, `digest`, `manifest.items`, `sum`, `COUNTS.values`, `sha256_file`, `ValueError`, `_indexed` 외 8개
- 반환 표현: `(manifest, samples)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`make_requests` — L276**
```python
make_requests(samples, candidate) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `samples.items`, `stage_seed`, `digest`, `requests.append`
- 반환 표현: `requests`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`candidate_identity` — L294**
```python
candidate_identity(model_path, candidate, cohort, training_manifest=None, training_manifest_sha256=None, checkpoint_metadata=None) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `recovery.model_identity`, `Path`, `any`, `ValueError`, `isinstance`, `re.fullmatch`, `sha256_file`, `read_json`, `manifest.get('sources', {}).get`, `manifest.get` 외 11개
- 반환 표현: `identity`; `identity | {'candidate': candidate, 'training_manifest_sha256': training_manifest_sha256, 'checkpoint_metadata_sha256': sha256_file(checkpoint_metadata), 'training_contract': contract,  …`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`infer` — L355**
```python
infer(data_root, model_path, base_path, output, candidate, deadline_utc, commit, training_manifest=None, training_manifest_sha256=None, checkpoint_metadata=None, backend_factory=None, slots=4) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `map`, `datetime.fromisoformat`, `deadline_utc.replace`, `type`, `ValueError`, `deadline.utcoffset`, `datetime.now`, `recovery.checkout`, `output.is_symlink`, `output.resolve().is_relative_to` 외 40개
- 반환 표현: `identity`
- 명시적 예외: `ValueError`, `TimeoutError`, `re-raise`; 전파되는 callee 예외는 별도.

**`_validate_response` — L451**
```python
_validate_response(row, request, candidate, run_kind) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `row.get`, `isinstance`, `any`, `type`, `ValueError`, `recovery.validate_completion`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`_read_completions` — L461**
```python
_read_completions(response_root, requested, candidate, run_kind) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(response_root).is_symlink`, `Path`, `ValueError`, `sorted`, `Path(response_root).glob`, `read_json`, `row.get`, `path.is_symlink`, `_validate_response`
- 반환 표현: `done`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`score_qa` — L475**
```python
score_qa(raw, sample, target, finish_reason) -> unannotated
```
- 기존 설명: A/B JSON answer extraction uses only the complete submission and original choices.
- 직접 호출: `str(raw).strip`, `str`, `submitted.startswith`, `terminal_json`, `value.get`, `set`, `type`, `str(answer).strip`, `math.isfinite`, `ValueError` 외 4개
- 반환 표현: `{'correct': False, 'parse_status': 'NO_PARSE', 'parsed_answer': None, 'answer_source': 'INVALID_WHOLE_JSON'}`; `result | {'answer_source': source}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`score_facts` — L504**
```python
score_facts(raw, target) -> unannotated
```
- 기존 설명: Ordered string facts, no coercion/tolerance; format and semantic field scores are separate.
- 직접 호출: `terminal_json`, `isinstance`, `set`, `observed.get`, `bool`, `all`, `type`, `len`, `sum`, `enumerate` 외 1개
- 반환 표현: `{'correct': exact, 'facts_exact': exact, 'json_valid': submitted is not None, 'schema_valid': bool(schema), 'title_exact': equal('title'), 'headers_exact': equal('headers'), 'rows_exact …`

**`_group` — L546**
```python
_group(cells, expected) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `len`, `sum`
- 반환 표현: `{'planned': expected, 'completed': len(cells), 'correct': sum((row['correct'] for row in cells)), 'accuracy': sum((row['correct'] for row in cells)) / expected if len(cells) == expected …`

**`score` — L553**
```python
score(data_root, gold_root, run_roots, output) -> unannotated
```
- 기존 설명: Explicit local CPU replay; incomplete denominators and expansion gates stay null.
- 직접 호출: `map`, `_new_destination`, `load_cohort`, `read_json`, `binding.get`, `sha256_file`, `ValueError`, `_indexed`, `read_rows`, `set` 외 30개
- 반환 표현: `summary`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L658**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_subparsers`, `commands.add_parser`, `preparer.add_argument`, `inference.add_argument`, `scorer.add_argument`, `vars`, `parser.parse_args`, `args.pop`, `prepare` 외 4개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.163 `fine_tuning/tabmwp_images.py`

[현재 소스](../../fine_tuning/tabmwp_images.py) · SHA256 `bfcaead34cf7d144a57971db0366bed3aaca1408707de5d219fe19a5836e9b23`

모듈의 기존 설명: Acquire only frozen TabMWP PNGs; byte integrity never grants training approval.

**`git_hash` — L26**
```python
git_hash(kind, data) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha1(f'{kind} {len(data)}\x00'.encode() + data).hexdigest`, `hashlib.sha1`, `f'{kind} {len(data)}\x00'.encode`, `len`
- 반환 표현: `hashlib.sha1(f'{kind} {len(data)}\x00'.encode() + data).hexdigest()`

**`download` — L30**
```python
download(url, limit) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `urllib.request.Request`, `urllib.request.urlopen`, `response.read`, `len`, `ValueError`
- 반환 표현: `data`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_tree` — L39**
```python
verify_tree(commit, tree) -> unannotated
```
- 기존 설명: Recompute every Git tree object, including the pinned repository root.
- 직접 호출: `commit.get`, `commit.get('tree', {}).get`, `tree.get`, `ValueError`, `defaultdict`, `entry.get`, `isinstance`, `path.startswith`, `any`, `path.split` 외 15개
- 반환 표현: `entries`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`selection` — L69**
```python
selection(prepared_root) -> unannotated
```
- 기존 설명: Bind immutable A/B records to the native fixed selection, without emitting gold.
- 직접 호출: `Path`, `json.loads`, `manifest_path.read_text`, `dict`, `hashed.pop`, `tabmwp.digest`, `manifest.get`, `ValueError`, `tabmwp.load_source`, `sha` 외 12개
- 반환 표현: `(paths, bound, expected)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`verify_png` — L116**
```python
verify_png(data, entry) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `entry.get`, `type`, `len`, `git_hash`, `data.startswith`, `ValueError`, `Image.open`, `io.BytesIO`, `image.verify`, `image.load` 외 8개
- 반환 표현: `{'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha1': entry['sha'], 'size': size, 'pixels_sha256': pixels}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`acquire` — L134**
```python
acquire(prepared_root, destination) -> unannotated
```
- 기존 설명: New bundle, or bounded resume of its exact incomplete acquisition; no overwrite.
- 직접 호출: `selection`, `Path`, `root.is_symlink`, `ValueError`, `root.exists`, `json.loads`, `(root / 'acquisition.json').read_text`, `sorted`, `root.glob`, `len` 외 26개
- 반환 표현: `receipt`
- 명시적 예외: `ValueError`, `FileExistsError`; 전파되는 callee 예외는 별도.

**`main` — L216**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `acquire`, `print`, `json.dumps`, `SystemExit`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `SystemExit`; 전파되는 callee 예외는 별도.

### D18.164 `fine_tuning/tabmwp_runpod.py`

[현재 소스](../../fine_tuning/tabmwp_runpod.py) · SHA256 `d19ee260d9672b8a8fd45454731a9d0965fc643a61c898b929dc6e1845f87a0d`

모듈의 기존 설명: Bounded fresh-F0 TabMWP A/B SFT and frozen external evaluation; no automatic expansion.

**`input_gate` — L20**
```python
input_gate(args) -> unannotated
```
- 기존 설명: Require actual closed diagnosis, reviewed CPU data and the exact new execution authority.
- 직접 호출: `read_json`, `setup.get`, `ValueError`, `Path(setup['model_path']).resolve`, `Path`, `model.is_relative_to`, `args.runtime_root.resolve`, `model_identity`, `diagnosis.get`, `re.fullmatch` 외 27개
- 반환 표현: `(model, manifests, bound)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`run` — L97**
```python
run(args, controller) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `input_gate`, `write_json`, `sha256_file`, `evaluate`, `contract`, `str`, `max`, `controller.remaining`, `stage`, `read_json` 외 7개
- 반환 표현: `{'status': 'AB_TRAINING_AND_EXTERNAL_INFERENCE_COMPLETED', 'training_executed': True, 'rows_per_variant': 2000, 'updates_per_variant': 125, 'external_actual_cells': 396, 'scoring_status …`
- 명시적 예외: `TimeoutError`, `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L169**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `checkout`, `os.environ.get`, `parser.error`, `path.resolve`, `canonical.is_relative_to`, `Path`, `effective_deadline` 외 10개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `re-raise`; 전파되는 callee 예외는 별도.

### D18.165 `fine_tuning/tabmwp_trial.py`

[현재 소스](../../fine_tuning/tabmwp_trial.py) · SHA256 `88af13cb8a56f59d44ebc8827c4e29ff33945bb74cb7a5009887eae72e1a3a82`

모듈의 기존 설명: Freeze a private TabMWP A/B trial; CPU approval is a separate reviewed action.

**`read_json` — L14**
```python
read_json(path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `Path(path).read_text`, `Path`
- 반환 표현: `json.loads(Path(path).read_text(encoding='utf-8'))`

**`unchanged` — L18**
```python
unchanged(bound) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `any`, `Path(name).is_file`, `Path`, `audit.sha`, `bound.items`, `ValueError`
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`native_bundle` — L23**
```python
native_bundle(prepared_root, image_root) -> unannotated
```
- 기존 설명: Replay native conversion and every pinned image, including dev/sealed images.
- 직접 호출: `Path`, `tabmwp_images.selection`, `read_json`, `ValueError`, `sorted`, `images.glob`, `receipt.get`, `tabmwp.digest`, `set`, `any` 외 16개
- 반환 표현: `(variants, bound, receipt_path)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`retention_bundle` — L73**
```python
retention_bundle(root) -> unannotated
```
- 기존 설명: Require the exact closed historical approval chain; no fallback to raw data.
- 직접 호출: `Path`, `prepare_training.closed_pilot`, `prepare_training.select_approved_subset`, `len`, `Counter`, `r.get`, `any`, `ValueError`, `str`, `frozen.items` 외 5개
- 반환 표현: `(selected, manifest, bound)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`pair_check` — L96**
```python
pair_check(a, b, retention_ids) -> unannotated
```
- 기존 설명: The only variant difference is native assistant target supervision.
- 직접 호출: `len`, `set`, `ValueError`, `zip`, `left.get`, `right.get`, `observed.add`, `left.items`, `right.items`, `json.loads` 외 1개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`contract` — L125**
```python
contract() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Path(__file__).with_name`, `Path`, `audit.training_input_contract`, `audit.sha`
- 반환 표현: `{'arm': 'FM', 'learning_rate': 1e-05, 'module_selection_basis': 'Historical G6 external development, not new MMMU outcomes', 'each_variant_starts_from': 'F0', 'dtype': 'bfloat16', 'quan …`

**`prepare` — L139**
```python
prepare(prepared_root, image_root, retention_root, output_root) -> unannotated
```
- 기존 설명: New proposals/registry only; all originals and historical approvals stay immutable.
- 직접 호출: `Path`, `output.exists`, `FileExistsError`, `native_bundle`, `retention_bundle`, `dict`, `ValueError`, `audit.sha`, `prepare_training.order`, `audit.check_splits` 외 16개
- 반환 표현: `manifest`
- 명시적 예외: `FileExistsError`, `ValueError`; 전파되는 callee 예외는 별도.

**`validate_trial` — L221**
```python
validate_trial(root) -> unannotated
```
- 기존 설명: Fail closed before any processor import or approval output.
- 직접 호출: `Path`, `read_json`, `dict`, `hashed.pop`, `tabmwp.digest`, `manifest.get`, `contract`, `ValueError`, `manifest['source_sha256'].items`, `audit.sha` 외 12개
- 반환 표현: `(variants, manifest)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`approve_cpu` — L261**
```python
approve_cpu(root, reviewed_registry, benchmark_index, model_path, review_path, output_root) -> unannotated
```
- 기존 설명: Explicit future CPU-only approval stage; preparation never calls this function.
- 직접 호출: `Path`, `output.exists`, `FileExistsError`, `validate_trial`, `read_json`, `audit.sha`, `any`, `review.get`, `required.items`, `ValueError` 외 24개
- 반환 표현: `report`
- 명시적 예외: `FileExistsError`, `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L343**
```python
main() -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_subparsers`, `commands.add_parser`, `prepare_parser.add_argument`, `approve_parser.add_argument`, `vars`, `parser.parse_args`, `args.pop`, `prepare`, `approve_cpu` 외 2개
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

### D18.166 `fine_tuning/train.py`

[현재 소스](../../fine_tuning/train.py) · SHA256 `a76400e234d19ea9f78392ce4dc1b44494341038d9450950cd7716fe590cd36e`

모듈의 기존 설명: Private BF16 selective SFT. No remote downloads or implicit GPU fallback.

**`file_hash` — L20**
```python
file_hash(path: Path) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256`, `path.open`, `iter`, `stream.read`, `digest.update`, `digest.hexdigest`
- 반환 표현: `digest.hexdigest()`

**`approved_records` — L28**
```python
approved_records(records_path: Path, manifest_path: Path, image_root: Path) -> tuple[list[dict], dict]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `json.loads`, `manifest_path.read_text`, `manifest.get`, `manifest.get('contamination_audit', {}).get`, `ValueError`, `file_hash`, `manifest.get('source_sha256', {}).get`, `Path(__file__).with_name`, `Path`, `records_path.read_text(encoding='utf-8').splitlines` 외 16개
- 반환 표현: `(records, manifest)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`validate_training_input_contract` — L67**
```python
validate_training_input_contract(manifest: dict, min_pixels: int, max_pixels: int, episode_pixels: int, sequence_budget: int, training_input_protocol: Path | None=None) -> dict
```
- 기존 설명: Bind CLI budgets to approved processor evidence before any model/GPU load.
- 직접 호출: `training_input_contract`, `tuple`, `manifest.get`, `proof.get`, `type`, `any`, `ValueError`
- 반환 표현: `contract`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**클래스 `AssistantCollator` — L91**

기저: `object`.

**`AssistantCollator.__init__` — L94**
```python
AssistantCollator.__init__(self, processor, image_root: Path, sequence_budget: int, episode_pixels: int) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: 직접 함수 호출 없음
- 반환 표현: 명시적 return 없음(None 또는 generator/예외 경로는 원문 확인)

**`AssistantCollator.__call__` — L100**
```python
AssistantCollator.__call__(self, record: dict) -> dict[str, torch.Tensor]
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `Image.open`, `images.append`, `original.convert`, `any`, `ValueError`, `content.append`, `self.processor.apply_chat_template`, `full.startswith`, `self.processor`, `torch.equal` 외 9개
- 반환 표현: `dict(inputs)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`probe_indices` — L149**
```python
probe_indices(numel: int, device: torch.device | str='cpu') -> torch.Tensor
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `min`, `torch.tensor`, `max`, `range`
- 반환 표현: `torch.tensor([i * (numel - 1) // max(1, count - 1) for i in range(count)], dtype=torch.long, device=device)`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`parameter_probes` — L158**
```python
parameter_probes(model, inventory: dict) -> dict
```
- 기존 설명: Bounded frozen sampling is explicit; trainable hashes are complete in export.
- 직접 호출: `model.named_parameters`, `parameter_group`, `max`, `len`, `parameter.detach().reshape`, `parameter.detach`, `probe_indices`, `flat.numel`, `tensor_sha256`
- 반환 표현: `selected`

**`optimizer_update` — L171**
```python
optimizer_update(model, optimizer, tokens: int, inventory: dict, probes: dict) -> dict
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `ValueError`, `dict`, `model.named_parameters`, `p.detach().flatten()[:1024].float().clone`, `p.detach().flatten()[:1024].float`, `p.detach().flatten`, `p.detach`, `parameter.grad.div_`, `float`, `parameter.grad.float().square().sum` 외 19개
- 반환 표현: `{'gradient_l2': norms, 'sampled_update_l1': deltas, 'frozen_sampled_checks': len(probes), 'loss_tokens': tokens}`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`order_hash` — L202**
```python
order_hash(ids: list[str]) -> str
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `hashlib.sha256(json.dumps(ids).encode()).hexdigest`, `hashlib.sha256`, `json.dumps(ids).encode`, `json.dumps`
- 반환 표현: `hashlib.sha256(json.dumps(ids).encode()).hexdigest()`

**`stop_for_budget` — L206**
```python
stop_for_budget(elapsed: float, budget_seconds: float | None, mean_update_seconds: float, checkpoint_reserve_seconds: float=120) -> bool
```
- 기존 설명: At update boundaries, leave time for another update and checkpoint export.
- 직접 호출: `max`
- 반환 표현: `budget_seconds is not None and elapsed + max(30, mean_update_seconds * 1.5) + checkpoint_reserve_seconds >= budget_seconds`

**`load_base` — L212**
```python
load_base(model_path: Path) -> unannotated
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `model_path.is_dir`, `ValueError`, `Qwen3VLForConditionalGeneration.from_pretrained`, `str`
- 반환 표현: `model`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

**`main` — L226**
```python
main() -> None
```
- 기존 설명: 명시적 docstring 없음. 아래 실제 서명·호출·반환과 원문을 기준으로 해석한다.
- 직접 호출: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `ValueError`, `load_checkpoint`, `load_base`, `print`, `json.dumps`, `str`, `sum` 외 66개
- 반환 표현: `None`
- 명시적 예외: `ValueError`; 전파되는 callee 예외는 별도.

### D18.T 현재 테스트 파일 색인

아래는 존재하는 테스트 파일과 함수 정의 수다. **이번 문서 작성에서 이 테스트들을 새로 실행했다는 뜻이 아니다.** 신규 제안 검사는 plan의 P/N 계약을 따른다.

| 파일 | 정의 수 | SHA256 |
|---|---:|---|
| [fine_tuning/notebook_rl/data_tests/__init__.py](../../fine_tuning/notebook_rl/data_tests/__init__.py) | 0 | `3deadd3b679ab40fb05935fe6e537a90e2c0ac6ee83a1f8011f4bbfed4902bf2` |
| [fine_tuning/notebook_rl/data_tests/test_ad_geometry.py](../../fine_tuning/notebook_rl/data_tests/test_ad_geometry.py) | 16 | `0ab42c051387f2a64cee1cb7fa71fed2f778bac8e9d4fb4bf3fe599677c7206d` |
| [fine_tuning/notebook_rl/data_tests/test_common.py](../../fine_tuning/notebook_rl/data_tests/test_common.py) | 7 | `a77b50d170d2a95df445f8e0fbf9151615c518a7d6c9235624049161474c4760` |
| [fine_tuning/notebook_rl/data_tests/test_gen_chart.py](../../fine_tuning/notebook_rl/data_tests/test_gen_chart.py) | 11 | `26963818cc58467c43ef31de1b3ddc64c4bc18c9dc3a7e54a8bee9c4246fd3a9` |
| [fine_tuning/notebook_rl/data_tests/test_gen_chem.py](../../fine_tuning/notebook_rl/data_tests/test_gen_chem.py) | 14 | `8c63dc9a3c32e27abe6c4a1e701ecff5e311352b0ecae899b1a96e62527f225e` |
| [fine_tuning/notebook_rl/data_tests/test_gen_circuit.py](../../fine_tuning/notebook_rl/data_tests/test_gen_circuit.py) | 16 | `f361ca2aa5f157becb4785ceefb25fb342a2756a80d2ff111aded1e3c2805733` |
| [fine_tuning/notebook_rl/data_tests/test_gen_graph.py](../../fine_tuning/notebook_rl/data_tests/test_gen_graph.py) | 18 | `abab4f4da98aaf30a56554cd80549ce676f3a3d38664532175294f33060469f4` |
| [fine_tuning/notebook_rl/data_tests/test_gen_graph_pair.py](../../fine_tuning/notebook_rl/data_tests/test_gen_graph_pair.py) | 6 | `ec1f179a7d2a594499be2a8f18fbfd51f84b2c255012f0b1b8579a5b22736d0e` |
| [fine_tuning/notebook_rl/data_tests/test_gen_music.py](../../fine_tuning/notebook_rl/data_tests/test_gen_music.py) | 11 | `225543d0197b60d4a9290726d783f1c21cd48e084398848c8cf689a0d3fb00d4` |
| [fine_tuning/notebook_rl/data_tests/test_gen_statics.py](../../fine_tuning/notebook_rl/data_tests/test_gen_statics.py) | 10 | `4b189c93de2294eb26c2320ce478c35a51cd186ed00f73d6a83843cb4ac5a004` |
| [fine_tuning/tests/test_ablation.py](../../fine_tuning/tests/test_ablation.py) | 11 | `f858d152f15d897c114f250d330d59f58f881b52d7b65a15283de082df5c8df5` |
| [fine_tuning/tests/test_bridge_evaluation.py](../../fine_tuning/tests/test_bridge_evaluation.py) | 12 | `d31ee7e9dbe065e1ebb00d994c8883d96d897a0299a77fa55ee1d252e5e0c9c7` |
| [fine_tuning/tests/test_bridge_experiment.py](../../fine_tuning/tests/test_bridge_experiment.py) | 12 | `f16447cda0721d832f5a70eb55e7677b9d51b99d83b1ef6981ee5b632c87b913` |
| [fine_tuning/tests/test_bridge_smoke.py](../../fine_tuning/tests/test_bridge_smoke.py) | 6 | `e26035d228cdcd5d3b4933495d8c8bea680a271b43725c875a1afe60c0fa52fb` |
| [fine_tuning/tests/test_budget.py](../../fine_tuning/tests/test_budget.py) | 11 | `7a64dc30299ca2d74f6ddc374490dc57149cea429f6f74efec0956328574ad37` |
| [fine_tuning/tests/test_circuit_bridge.py](../../fine_tuning/tests/test_circuit_bridge.py) | 7 | `87dba0d0ffb6b49011f409ef12a4e56b2c9411e0c9441d7cd7ee6cc699975859` |
| [fine_tuning/tests/test_circuit_evaluation.py](../../fine_tuning/tests/test_circuit_evaluation.py) | 10 | `440de5db78e8d1fe7b8692727877a0e473a8a7ce5f2b4920546426a57de23b3d` |
| [fine_tuning/tests/test_circuit_scoring.py](../../fine_tuning/tests/test_circuit_scoring.py) | 16 | `0bb136bbff6059dd3cfe087c85ae093197aedaf75b1aee07d319c18aacbd1cb7` |
| [fine_tuning/tests/test_control.py](../../fine_tuning/tests/test_control.py) | 7 | `024c76132b649f7e3574c6e50a41ab8410c7139698350893b52466d4f5c3dd12` |
| [fine_tuning/tests/test_controller.py](../../fine_tuning/tests/test_controller.py) | 14 | `54442ee6123acd9578a30d9d52c15fcbd42e004ea425f18715f0ec6e8b2abbda` |
| [fine_tuning/tests/test_data_audit.py](../../fine_tuning/tests/test_data_audit.py) | 22 | `a1373897cee8f4870f6a716b3cfee26609645fca9b87b171e6232366ecea2e93` |
| [fine_tuning/tests/test_evaluate.py](../../fine_tuning/tests/test_evaluate.py) | 14 | `db94f037bb04678649a8cb18e6c29a1b74c77bf5fd6f55d9803fb38e15962676` |
| [fine_tuning/tests/test_evaluate_cli.py](../../fine_tuning/tests/test_evaluate_cli.py) | 2 | `41f5f80aed0e6528fd57f3d376cec1db34a1f125c12611a5274b3b1b4f4d8ece` |
| [fine_tuning/tests/test_graph_bridge.py](../../fine_tuning/tests/test_graph_bridge.py) | 12 | `180830db3dcaca7282baad535934fa61979fc3681a57cb07a01cca268d7ffed0` |
| [fine_tuning/tests/test_notebook_retained_signal.py](../../fine_tuning/tests/test_notebook_retained_signal.py) | 18 | `8d11a30e32b899337001ce0101286072aecddd77f81f5921aa6831baf79d0a03` |
| [fine_tuning/tests/test_notebook_rl_adjudication.py](../../fine_tuning/tests/test_notebook_rl_adjudication.py) | 16 | `7783e24cbfbfa533139cf9271db139229de7defcd76b30359e1e1adfb2f334b2` |
| [fine_tuning/tests/test_notebook_rl_all_pair_credit.py](../../fine_tuning/tests/test_notebook_rl_all_pair_credit.py) | 8 | `cf251abe3b1847eaf1fc0c0e4e109881935af96dd2f091b5849ae6897e5184a4` |
| [fine_tuning/tests/test_notebook_rl_answers.py](../../fine_tuning/tests/test_notebook_rl_answers.py) | 14 | `9d9cb46dd1293eb78cdc966cb695e1f984cbc59b966f7c208580b7ea8d7f7543` |
| [fine_tuning/tests/test_notebook_rl_approval.py](../../fine_tuning/tests/test_notebook_rl_approval.py) | 68 | `667585cdbff67c69714f9e3204c460b7933bc18ac9584612caae74ea59f010eb` |
| [fine_tuning/tests/test_notebook_rl_assessment_admission.py](../../fine_tuning/tests/test_notebook_rl_assessment_admission.py) | 21 | `7377a7f8accc19eaa6ac3e9b0cc108b65e3685570358609c0471320fc1afef08` |
| [fine_tuning/tests/test_notebook_rl_assessment_credit.py](../../fine_tuning/tests/test_notebook_rl_assessment_credit.py) | 27 | `a8af49dcc1836180088e9250cfc407acd781d2eaf17396aaed7586d07922e278` |
| [fine_tuning/tests/test_notebook_rl_assessment_profile.py](../../fine_tuning/tests/test_notebook_rl_assessment_profile.py) | 12 | `ed162d7c6a738cc2a309d3966771c2d71bf5cd963ba3411fbaa6bc740bfa864f` |
| [fine_tuning/tests/test_notebook_rl_assessment_update.py](../../fine_tuning/tests/test_notebook_rl_assessment_update.py) | 32 | `690663bedb2956e2bb4f3894a2c9818b8b769425044e0284e51ff83b00433a09` |
| [fine_tuning/tests/test_notebook_rl_attribution.py](../../fine_tuning/tests/test_notebook_rl_attribution.py) | 11 | `466ba9bc3e78c43d0ed399d6bd797dd12b347f4e81f431d7a7c896e922028212` |
| [fine_tuning/tests/test_notebook_rl_batches.py](../../fine_tuning/tests/test_notebook_rl_batches.py) | 10 | `d7c50cc86ebfbdfed2d0424548f9dbef360c6e0c1cafba2c0ba4bbb8dc4efdcb` |
| [fine_tuning/tests/test_notebook_rl_build_exclusion_index.py](../../fine_tuning/tests/test_notebook_rl_build_exclusion_index.py) | 13 | `be02179c5a78d8a170ad68fc0156df7c16c67cd5f1a3c35de6264dc4b68a935a` |
| [fine_tuning/tests/test_notebook_rl_checkpoint.py](../../fine_tuning/tests/test_notebook_rl_checkpoint.py) | 24 | `7b5fdf5d9fe8b74159c6242d53970c84892919c26b56a7bf0b270f4e1b33a8a8` |
| [fine_tuning/tests/test_notebook_rl_circuit_notebook.py](../../fine_tuning/tests/test_notebook_rl_circuit_notebook.py) | 11 | `e3670eca41bd0414f2a5b6c797ed19a0308a764dfa2032b87f47ede0bc88b1e9` |
| [fine_tuning/tests/test_notebook_rl_circuit_trace.py](../../fine_tuning/tests/test_notebook_rl_circuit_trace.py) | 20 | `4585de0f1fa7e39ea3982f9b6a0656bf1aa215b14e759c05ac85813f7a4272e2` |
| [fine_tuning/tests/test_notebook_rl_cli.py](../../fine_tuning/tests/test_notebook_rl_cli.py) | 10 | `75322d0debc4c3b9802eaec3110131fd613a42c52546ce79effbe4d26355092c` |
| [fine_tuning/tests/test_notebook_rl_completion_observer.py](../../fine_tuning/tests/test_notebook_rl_completion_observer.py) | 10 | `e5897f39ba20917cd5edd4d6c596705da605db3ac0c1b361f3993b8b40ba4629` |
| [fine_tuning/tests/test_notebook_rl_composite_notebook.py](../../fine_tuning/tests/test_notebook_rl_composite_notebook.py) | 13 | `3b2adf9ba7b280c70987bc3e5159eca06594908fb38a4b811e8bb38fa41fdfe3` |
| [fine_tuning/tests/test_notebook_rl_contamination.py](../../fine_tuning/tests/test_notebook_rl_contamination.py) | 40 | `d316bf562167b412bed832dadddfb83ac5e064a3dc181fd2a6a3feb99bfa615c` |
| [fine_tuning/tests/test_notebook_rl_dataset.py](../../fine_tuning/tests/test_notebook_rl_dataset.py) | 12 | `dd44a58cefcab92ed12c4500f80527d87d7df5b3d5213920e3a13740c707eb30` |
| [fine_tuning/tests/test_notebook_rl_determinism.py](../../fine_tuning/tests/test_notebook_rl_determinism.py) | 5 | `3dbbf7644e25a314c1f49046441245933c2c28d678b2bf5aaec275bd68c582a6` |
| [fine_tuning/tests/test_notebook_rl_empty_synthesis_credit.py](../../fine_tuning/tests/test_notebook_rl_empty_synthesis_credit.py) | 6 | `5830816e03ce52e8fe40697111af2d421b32eb7f746dd568e46e3c290e69a2ea` |
| [fine_tuning/tests/test_notebook_rl_engines.py](../../fine_tuning/tests/test_notebook_rl_engines.py) | 39 | `4560e0709236d13949ff92d92c0773b5eecd72e1633b96e2360510ed24d2a93b` |
| [fine_tuning/tests/test_notebook_rl_episode_notebook.py](../../fine_tuning/tests/test_notebook_rl_episode_notebook.py) | 41 | `6ac6186b5ae1097511137d3280ddbd1b7ef5683a8d34c1f1f69bcd067f4541ea` |
| [fine_tuning/tests/test_notebook_rl_episode_retry.py](../../fine_tuning/tests/test_notebook_rl_episode_retry.py) | 17 | `6eaa45ed430c0d49196f80a6132af5d0df7b68908651501fe20e238b34a16c1d` |
| [fine_tuning/tests/test_notebook_rl_eval_external.py](../../fine_tuning/tests/test_notebook_rl_eval_external.py) | 23 | `8445ab861026ceed70ec6beebd3e336af5f05d2d654ea449d9f812b833c39710` |
| [fine_tuning/tests/test_notebook_rl_evaluation_audit.py](../../fine_tuning/tests/test_notebook_rl_evaluation_audit.py) | 14 | `29d98a4773eda8a60902f9db4bc30014c982a8614edd96b13c2b905e66530145` |
| [fine_tuning/tests/test_notebook_rl_exclusion_bundle.py](../../fine_tuning/tests/test_notebook_rl_exclusion_bundle.py) | 8 | `ab3b6386ae5f3dd85c2ec527acf597db8cba6681ccf84aadbc26d26719efb0e7` |
| [fine_tuning/tests/test_notebook_rl_exclusion_calibration.py](../../fine_tuning/tests/test_notebook_rl_exclusion_calibration.py) | 8 | `372a79e2752280bd606387033a554cbac6e08640bb5fbbb816631350265b4e03` |
| [fine_tuning/tests/test_notebook_rl_exclusion_engines.py](../../fine_tuning/tests/test_notebook_rl_exclusion_engines.py) | 17 | `90c4f2461ebe526c0744f495d7b9d7a8d135def8ab0ee45c54cf6ae45f5d6a82` |
| [fine_tuning/tests/test_notebook_rl_fact_context_probe.py](../../fine_tuning/tests/test_notebook_rl_fact_context_probe.py) | 24 | `c85d5188ec2d153efff77e4b64179069a8a295832f961a42c17a9b0dbffe3547` |
| [fine_tuning/tests/test_notebook_rl_fact_credit.py](../../fine_tuning/tests/test_notebook_rl_fact_credit.py) | 11 | `588f34ec3ef201cadd7619a9f75f0432add7f2e00478821912c9fb54512dae79` |
| [fine_tuning/tests/test_notebook_rl_fact_credit_loss.py](../../fine_tuning/tests/test_notebook_rl_fact_credit_loss.py) | 14 | `781e0028955981a856968e5efc21b4a06fff565639f569d6279419c6de7dc35e` |
| [fine_tuning/tests/test_notebook_rl_fact_credit_update.py](../../fine_tuning/tests/test_notebook_rl_fact_credit_update.py) | 7 | `f0e56c9d7968aaf41fd4ecf55ed2b4a16daec2bbecde855bff32109f3ec105cb` |
| [fine_tuning/tests/test_notebook_rl_fact_feedback.py](../../fine_tuning/tests/test_notebook_rl_fact_feedback.py) | 11 | `583a87d6c66922be2b29c657fd9b37eb493ecdfa96b3e65e372e4bf0a7281263` |
| [fine_tuning/tests/test_notebook_rl_fresh_observation.py](../../fine_tuning/tests/test_notebook_rl_fresh_observation.py) | 27 | `337aa22370f54f24bac23a733d56de22b5bb87edc8bf11008468af904881fcd3` |
| [fine_tuning/tests/test_notebook_rl_gate_cache.py](../../fine_tuning/tests/test_notebook_rl_gate_cache.py) | 16 | `087668831c22ac0a5410df7c3aa086922d6828005e741b285ccd0b06d5bf3b4d` |
| [fine_tuning/tests/test_notebook_rl_gate_checkpoint.py](../../fine_tuning/tests/test_notebook_rl_gate_checkpoint.py) | 20 | `60aadf955b43c48a5ef14b1bb348d3e21da23711f4721468e3704fba0de0486c` |
| [fine_tuning/tests/test_notebook_rl_gate_gradients.py](../../fine_tuning/tests/test_notebook_rl_gate_gradients.py) | 18 | `03e225e589332cef7a47ca04a6c900e69d897f203d7a9ef54e872d48e6111ff4` |
| [fine_tuning/tests/test_notebook_rl_gate_guards.py](../../fine_tuning/tests/test_notebook_rl_gate_guards.py) | 17 | `112ecd5c6fd970436d7695a8e7accac610d9a25ecfa0707fd29b088b1a559579` |
| [fine_tuning/tests/test_notebook_rl_gate_images.py](../../fine_tuning/tests/test_notebook_rl_gate_images.py) | 15 | `29b00a0a9c5924f8f4bd95203962595e12541fc643c2892a9063ca2f02026660` |
| [fine_tuning/tests/test_notebook_rl_gate_inputs.py](../../fine_tuning/tests/test_notebook_rl_gate_inputs.py) | 13 | `dd1ae5239c3adf23e84b1b30035715d085253754fa4c8a7662bc6273536ec876` |
| [fine_tuning/tests/test_notebook_rl_gate_logprobs.py](../../fine_tuning/tests/test_notebook_rl_gate_logprobs.py) | 14 | `c8c23e551f50311612a17a8d65bc37bc05a0a143523afe9541cf085c0be31a93` |
| [fine_tuning/tests/test_notebook_rl_gate_numerics.py](../../fine_tuning/tests/test_notebook_rl_gate_numerics.py) | 10 | `3cbac4a654e553c4b24e69c3f39df2e0b0dd415000b2e20a014e7be892bf4b47` |
| [fine_tuning/tests/test_notebook_rl_gate_reference.py](../../fine_tuning/tests/test_notebook_rl_gate_reference.py) | 14 | `fbe96dd23ae4355e676053fe3cba3aa1b18f3b98483efde3533d5b1ab8a8964a` |
| [fine_tuning/tests/test_notebook_rl_gate_sequence.py](../../fine_tuning/tests/test_notebook_rl_gate_sequence.py) | 9 | `9f20ac00f6d61eb99b409a2c10e9bad9b824dd9e967ae13ffe397455768dd407` |
| [fine_tuning/tests/test_notebook_rl_gate_session.py](../../fine_tuning/tests/test_notebook_rl_gate_session.py) | 20 | `0c66d6d490be9f0e47739f667bcbf7d57df0e488fcd8d46d5a0fca63c0c1478e` |
| [fine_tuning/tests/test_notebook_rl_gate_snapshot.py](../../fine_tuning/tests/test_notebook_rl_gate_snapshot.py) | 17 | `a483e9a35f7c4c8f17c882f72b70168205d42457d9e0bd131469b358d3621407` |
| [fine_tuning/tests/test_notebook_rl_gate_throughput.py](../../fine_tuning/tests/test_notebook_rl_gate_throughput.py) | 31 | `2675d27caa87c7cb399f3020eec58f575035acfdea89ed3b0eedd74560e79621` |
| [fine_tuning/tests/test_notebook_rl_gate_tokens.py](../../fine_tuning/tests/test_notebook_rl_gate_tokens.py) | 18 | `f81ace114894006099bd42bf7b73ff3e1d0bcc88e69c8f726cf4d202a086741a` |
| [fine_tuning/tests/test_notebook_rl_gate_update.py](../../fine_tuning/tests/test_notebook_rl_gate_update.py) | 15 | `958be66c87a0d797b28f391c582f1c822de2cb22c68b28351f58ecef492d5fb5` |
| [fine_tuning/tests/test_notebook_rl_gate_weights.py](../../fine_tuning/tests/test_notebook_rl_gate_weights.py) | 19 | `e13f79972e0ccd95d50033f3c2c7bfdbf7fb6c91a0b1e89d26cc810584fef071` |
| [fine_tuning/tests/test_notebook_rl_gpu_gates.py](../../fine_tuning/tests/test_notebook_rl_gpu_gates.py) | 16 | `e1691c2d9c91f2d2f3a265380980b9d05cff8e88334d35492c85d4ee0ffdce80` |
| [fine_tuning/tests/test_notebook_rl_guards.py](../../fine_tuning/tests/test_notebook_rl_guards.py) | 14 | `82afdf3be166f14ca263366cc2186135f4a887ddbed0039865fb50bed09256ec` |
| [fine_tuning/tests/test_notebook_rl_historical_verifier.py](../../fine_tuning/tests/test_notebook_rl_historical_verifier.py) | 15 | `101e382691be3c9c19da04b32c063b65fb92b3e471ee228160160f37bfca333b` |
| [fine_tuning/tests/test_notebook_rl_launch.py](../../fine_tuning/tests/test_notebook_rl_launch.py) | 12 | `48c10a639cdb9ba5fdfe63ead3d0855465c4de7d84a89cdb991def2a5cfcfe17` |
| [fine_tuning/tests/test_notebook_rl_local_calibrate.py](../../fine_tuning/tests/test_notebook_rl_local_calibrate.py) | 26 | `66db4193cb1f02a9ed149308c60c263841dc78308a7fdadd0b7543b54562c7b7` |
| [fine_tuning/tests/test_notebook_rl_local_control.py](../../fine_tuning/tests/test_notebook_rl_local_control.py) | 5 | `4eb2e87094f859aec7fe4c36f8bf51e969c0d63c99e3cbecab0e2e2121abc01f` |
| [fine_tuning/tests/test_notebook_rl_local_data.py](../../fine_tuning/tests/test_notebook_rl_local_data.py) | 18 | `5eabcf3b2d8cbad0ba14db0da2a33584b0710c4629f6bfbdead70f2844e17a77` |
| [fine_tuning/tests/test_notebook_rl_local_eval.py](../../fine_tuning/tests/test_notebook_rl_local_eval.py) | 61 | `0f17145903a99fb7aab76f83c211a31881cf22bced04178080257504bcf5b6af` |
| [fine_tuning/tests/test_notebook_rl_local_external.py](../../fine_tuning/tests/test_notebook_rl_local_external.py) | 14 | `22bab68ca904f440f127800bea410c29cd1c45b734eb3a3b72901f853e6947b3` |
| [fine_tuning/tests/test_notebook_rl_local_ladder.py](../../fine_tuning/tests/test_notebook_rl_local_ladder.py) | 15 | `fbdc3ee0c1f04bbc3837cb96e2b103916faa7dddf846ff4de5c171569dd91b29` |
| [fine_tuning/tests/test_notebook_rl_local_ladder_gate.py](../../fine_tuning/tests/test_notebook_rl_local_ladder_gate.py) | 11 | `3f8ca305f4bb7c83a75b5087b38e5883eb2b349e5ad5df6642bdb02ebd92fe86` |
| [fine_tuning/tests/test_notebook_rl_local_ladder_generation.py](../../fine_tuning/tests/test_notebook_rl_local_ladder_generation.py) | 26 | `ea052c9e1cf2862d7f1a5b5a1dbed0b594b1d0eff45bd95701631cd87738130c` |
| [fine_tuning/tests/test_notebook_rl_local_ladder_recovery.py](../../fine_tuning/tests/test_notebook_rl_local_ladder_recovery.py) | 34 | `4160dcc00529b61095c989390a3913891f21a60743ebeaf6484fd09df258c83a` |
| [fine_tuning/tests/test_notebook_rl_local_ladder_score.py](../../fine_tuning/tests/test_notebook_rl_local_ladder_score.py) | 16 | `8ab733f385ae19763f9605e9acc1737031bffec8dd11fdc1da47340c1edb53d7` |
| [fine_tuning/tests/test_notebook_rl_local_pipeline.py](../../fine_tuning/tests/test_notebook_rl_local_pipeline.py) | 14 | `25b05689e03b27ba58893478102c592b5b370db612791b5f39dabaddfb65530d` |
| [fine_tuning/tests/test_notebook_rl_local_profile.py](../../fine_tuning/tests/test_notebook_rl_local_profile.py) | 11 | `bbb8c68345a68b7b950d0b051b4b06246297445597503967b3bbd887a78f2608` |
| [fine_tuning/tests/test_notebook_rl_local_run.py](../../fine_tuning/tests/test_notebook_rl_local_run.py) | 14 | `175c95ad1cedd27bb133662bc4dddac63e8daa68452f5ed2ca5a980c026b085d` |
| [fine_tuning/tests/test_notebook_rl_local_train.py](../../fine_tuning/tests/test_notebook_rl_local_train.py) | 54 | `e9916a989a7fd20678f32c76bbe761a4fe8c83478bf3db91549dbea0d96a076d` |
| [fine_tuning/tests/test_notebook_rl_losses.py](../../fine_tuning/tests/test_notebook_rl_losses.py) | 16 | `3d64ccc308257b9da603c416182d34e45b82b574af4f5b35f372b469a5ca08b3` |
| [fine_tuning/tests/test_notebook_rl_measure.py](../../fine_tuning/tests/test_notebook_rl_measure.py) | 13 | `879e7bda580db8c0f44ad730488642eeb1e27df8651c8e577f850745d872c1e7` |
| [fine_tuning/tests/test_notebook_rl_metrics.py](../../fine_tuning/tests/test_notebook_rl_metrics.py) | 7 | `cc89a3872d13b460d96b039a68685471c134a31729b206bdd4634cd59eb2bbd9` |
| [fine_tuning/tests/test_notebook_rl_monitor.py](../../fine_tuning/tests/test_notebook_rl_monitor.py) | 26 | `aada2c590194d4441bc0e3661735353cac084e3edd318e82bd22744fbd1ce475` |
| [fine_tuning/tests/test_notebook_rl_native_commit_gate.py](../../fine_tuning/tests/test_notebook_rl_native_commit_gate.py) | 20 | `0f7cb2c197dc8b83cace0e8ab47e5ea0341ceae2de87a6bb5031bfedec09e27a` |
| [fine_tuning/tests/test_notebook_rl_native_exploration.py](../../fine_tuning/tests/test_notebook_rl_native_exploration.py) | 23 | `46999b5283a0707fb701b2f8bc7c9777fb4735e2f9695da13eff3e2fb4086a85` |
| [fine_tuning/tests/test_notebook_rl_native_relations.py](../../fine_tuning/tests/test_notebook_rl_native_relations.py) | 14 | `86cb5d138b5df2be95f76e028ded551c1fbf265ea9fe25635431507a45a529da` |
| [fine_tuning/tests/test_notebook_rl_native_relationship_gradient.py](../../fine_tuning/tests/test_notebook_rl_native_relationship_gradient.py) | 16 | `d2414e984c63abe3487cfc5e3dead43dc400d4dfebe145bc9cbcb8e1ca34c670` |
| [fine_tuning/tests/test_notebook_rl_notebook_draft_probe.py](../../fine_tuning/tests/test_notebook_rl_notebook_draft_probe.py) | 9 | `84b63ae9ac9527b8bdb8d9798ae3fc81d6523443dab144e0d6a2aacc344f2386` |
| [fine_tuning/tests/test_notebook_rl_notebook_layout_probe.py](../../fine_tuning/tests/test_notebook_rl_notebook_layout_probe.py) | 16 | `b4ca7b6e4d302e598e7a8310885a7a399404e882e45bf39f039c743fb2800ec4` |
| [fine_tuning/tests/test_notebook_rl_notebook_learning.py](../../fine_tuning/tests/test_notebook_rl_notebook_learning.py) | 38 | `8dd86cf49562251ea8a6a1720225787b506407531cadcc1d20f1b5150c805742` |
| [fine_tuning/tests/test_notebook_rl_notebook_objective.py](../../fine_tuning/tests/test_notebook_rl_notebook_objective.py) | 25 | `59d5b05d83d6470d4494d77669b78c23041dbfe1546e0a1bb64683189f2f72d9` |
| [fine_tuning/tests/test_notebook_rl_notebook_objective_update.py](../../fine_tuning/tests/test_notebook_rl_notebook_objective_update.py) | 14 | `17e904e930de57f801ab9901b5d8f77eacfae99d8da7cd58ca97652d3ef0ea57` |
| [fine_tuning/tests/test_notebook_rl_notebook_synthesis.py](../../fine_tuning/tests/test_notebook_rl_notebook_synthesis.py) | 38 | `8ffcc1953378a6328ce7572388ab6cdd024e026f5b57ae10e5b2ab712ac7a493` |
| [fine_tuning/tests/test_notebook_rl_observations.py](../../fine_tuning/tests/test_notebook_rl_observations.py) | 34 | `6abc04b739a65940d5a2c0ab7041a265118f1d7e4dc2f1b99ab68ec1ddaefb01` |
| [fine_tuning/tests/test_notebook_rl_pages_collection.py](../../fine_tuning/tests/test_notebook_rl_pages_collection.py) | 14 | `e590e0af8faad5326b9b9137afc9299f328f37f56d1f00b4d9a4a762bfc111eb` |
| [fine_tuning/tests/test_notebook_rl_pages_credit.py](../../fine_tuning/tests/test_notebook_rl_pages_credit.py) | 9 | `e09a37b2ab1ae06cc1f29748fe0b011fcfd9d33c68ef9f6812119c2177432e0c` |
| [fine_tuning/tests/test_notebook_rl_pages_profile.py](../../fine_tuning/tests/test_notebook_rl_pages_profile.py) | 11 | `a5f33d82d956884ae2fb723f2cfff31919848b6e496c2563ebcc9bf7eb7353a9` |
| [fine_tuning/tests/test_notebook_rl_pages_trained_probe.py](../../fine_tuning/tests/test_notebook_rl_pages_trained_probe.py) | 13 | `5915e1e3308e5b0340f5ba31d27168e5af28c511568f2f81c2b2f532e1fbd683` |
| [fine_tuning/tests/test_notebook_rl_pages_updates.py](../../fine_tuning/tests/test_notebook_rl_pages_updates.py) | 12 | `94c85df3af70b42a9b85e9c83beadf45174c5695f32de5c9ee11ced1bc7dd474` |
| [fine_tuning/tests/test_notebook_rl_pair_action.py](../../fine_tuning/tests/test_notebook_rl_pair_action.py) | 12 | `4fdd00471e2c63be6de6482c9c774a1461ea30814c0173edb68b08395b4844e9` |
| [fine_tuning/tests/test_notebook_rl_pair_action_credit.py](../../fine_tuning/tests/test_notebook_rl_pair_action_credit.py) | 15 | `1499366ee84df01340b6ba9a2d7ae77bdad43c0f295e4d71d33d60ac27f4bf44` |
| [fine_tuning/tests/test_notebook_rl_pair_action_integration.py](../../fine_tuning/tests/test_notebook_rl_pair_action_integration.py) | 16 | `bafe995699d64695f13609f19a2982012d5cd731935d45acc373b800d8957c99` |
| [fine_tuning/tests/test_notebook_rl_pair_batch_validation.py](../../fine_tuning/tests/test_notebook_rl_pair_batch_validation.py) | 10 | `079812d6d004c7e797a5f39b4ca5222b8e5bc35fb26773ecd4c86a3697ad5221` |
| [fine_tuning/tests/test_notebook_rl_pair_drafts.py](../../fine_tuning/tests/test_notebook_rl_pair_drafts.py) | 15 | `544a7caaa7b92fec30e40c634974ce362e779bc7028c8d10052c0a10a7469e08` |
| [fine_tuning/tests/test_notebook_rl_pair_probe.py](../../fine_tuning/tests/test_notebook_rl_pair_probe.py) | 12 | `2ef2c071b9ccc654bd2948097c8c65f0c0176f1b777c84faca3a0661a5dd537c` |
| [fine_tuning/tests/test_notebook_rl_pair_probe_integration.py](../../fine_tuning/tests/test_notebook_rl_pair_probe_integration.py) | 11 | `78bfe571cbf3115ab92508596000a6224b59610d323731365e3d470dfefcab7c` |
| [fine_tuning/tests/test_notebook_rl_pair_to_whole_probe.py](../../fine_tuning/tests/test_notebook_rl_pair_to_whole_probe.py) | 12 | `57963d343c5ce75693baee73dfed11e570a0ec3c019e8d8b450f02146f6642b5` |
| [fine_tuning/tests/test_notebook_rl_paired_credit.py](../../fine_tuning/tests/test_notebook_rl_paired_credit.py) | 19 | `b52342661218144bacd9f0d4ac9eb110afecc81aceba87bf0ec2142c401918ca` |
| [fine_tuning/tests/test_notebook_rl_paired_integration.py](../../fine_tuning/tests/test_notebook_rl_paired_integration.py) | 13 | `ea2cd7789eecef796fe6d9580146d635b4138de0f727883caf8b225e48ef1de8` |
| [fine_tuning/tests/test_notebook_rl_paired_notebook.py](../../fine_tuning/tests/test_notebook_rl_paired_notebook.py) | 18 | `d05f2193fc9f51161a7db159b2b023e266cc3b0d6b152740e3340f7e9cbd134d` |
| [fine_tuning/tests/test_notebook_rl_paired_profile.py](../../fine_tuning/tests/test_notebook_rl_paired_profile.py) | 11 | `edd716c2563a4d36096243efc670d3fc4331364fa6f26b61996d33d58daab3e4` |
| [fine_tuning/tests/test_notebook_rl_partial_integration.py](../../fine_tuning/tests/test_notebook_rl_partial_integration.py) | 9 | `f96534c324ef577111b3faf32819dd352c5cb5e96ae462fb9c00e0811468f396` |
| [fine_tuning/tests/test_notebook_rl_partial_notebook.py](../../fine_tuning/tests/test_notebook_rl_partial_notebook.py) | 10 | `116a4e370cb82bf1fe65a8df58c63a3144e900fcaffc89a5c40ed8b997633fe8` |
| [fine_tuning/tests/test_notebook_rl_partial_profile.py](../../fine_tuning/tests/test_notebook_rl_partial_profile.py) | 11 | `e1022e08d0f58c1fd3df0df6f61b41f726ccdea3becc7ed797b2b3ba247ea491` |
| [fine_tuning/tests/test_notebook_rl_phase_a.py](../../fine_tuning/tests/test_notebook_rl_phase_a.py) | 21 | `db7b6865ac649959ba3ae549b6702b35ec730d720a8c2ef2a3a608caaff92ec7` |
| [fine_tuning/tests/test_notebook_rl_policy.py](../../fine_tuning/tests/test_notebook_rl_policy.py) | 12 | `f8a003200e5b934ab0b62d74eb5a4948f7b405f06669729e47ccfcfb66cc29ea` |
| [fine_tuning/tests/test_notebook_rl_portability.py](../../fine_tuning/tests/test_notebook_rl_portability.py) | 4 | `67c94a6b22e202e4360f06ff4b12204d3ffc4751106f5551becb2f8bdff4a9ed` |
| [fine_tuning/tests/test_notebook_rl_prompts.py](../../fine_tuning/tests/test_notebook_rl_prompts.py) | 13 | `148d09c4303f8353d546bc5e316a6ad58097bf0fc5f3f662248220c81b8ffb04` |
| [fine_tuning/tests/test_notebook_rl_public_coverage.py](../../fine_tuning/tests/test_notebook_rl_public_coverage.py) | 6 | `c2e5a9ff7d074214745f440f2283666641367d71c49c04fc2c5f8b4bfa863597` |
| [fine_tuning/tests/test_notebook_rl_quantized_policy.py](../../fine_tuning/tests/test_notebook_rl_quantized_policy.py) | 34 | `e878dd7a370c01c00ccf5946dfaa882d7881e9b03b5ee5d9474265ffbe04d1b5` |
| [fine_tuning/tests/test_notebook_rl_reasoning_trace.py](../../fine_tuning/tests/test_notebook_rl_reasoning_trace.py) | 46 | `9aa50d3674aa6823d2be3c0078287d96e379f473ab94be67cc9f4ef0ffa47e09` |
| [fine_tuning/tests/test_notebook_rl_reference_ladder_evaluation.py](../../fine_tuning/tests/test_notebook_rl_reference_ladder_evaluation.py) | 10 | `e89439a58976f27aa1c1b632e6c8b6a1cea16e80ca5de8fdd0e20469a3dd79e3` |
| [fine_tuning/tests/test_notebook_rl_referenced_credit.py](../../fine_tuning/tests/test_notebook_rl_referenced_credit.py) | 17 | `802d65cda8b8ae4da5a4cafac5750b821fe471c868622ab8a52a330967ad80a3` |
| [fine_tuning/tests/test_notebook_rl_referenced_integration.py](../../fine_tuning/tests/test_notebook_rl_referenced_integration.py) | 11 | `cb1153dd6eec8dd10369602b64618b69ff2d30c96f9d3408edfa3e55beb806a4` |
| [fine_tuning/tests/test_notebook_rl_referenced_notebook.py](../../fine_tuning/tests/test_notebook_rl_referenced_notebook.py) | 13 | `c9f3aa4dae31fe8449b41a8ef3f35551316d3167c7a1c191678457620b369ba4` |
| [fine_tuning/tests/test_notebook_rl_referenced_profile.py](../../fine_tuning/tests/test_notebook_rl_referenced_profile.py) | 8 | `ab328010c22cf109e4882f2c2d4a4831a098e10caa7f4222b7b9dce39141e5db` |
| [fine_tuning/tests/test_notebook_rl_relationship_probe.py](../../fine_tuning/tests/test_notebook_rl_relationship_probe.py) | 21 | `e9bf09ad9155451f17843cd329b4e5e463329a9d7d5958bc68c3c1018b0800cc` |
| [fine_tuning/tests/test_notebook_rl_relationship_probe_runtime.py](../../fine_tuning/tests/test_notebook_rl_relationship_probe_runtime.py) | 7 | `a10109735721e9e7961ed2641b72cb2e1119584b925e61a98f9eb81a300b6adc` |
| [fine_tuning/tests/test_notebook_rl_rollout.py](../../fine_tuning/tests/test_notebook_rl_rollout.py) | 26 | `eca221c1d525c454ab3473a041b75474df8f7ce099ec449a4a215f813cf4e94f` |
| [fine_tuning/tests/test_notebook_rl_schema.py](../../fine_tuning/tests/test_notebook_rl_schema.py) | 17 | `edd98378df1cf4fc773014dcd19636502f376cb47b8e09d0db24b5ed5f711c51` |
| [fine_tuning/tests/test_notebook_rl_shared_policy_engine.py](../../fine_tuning/tests/test_notebook_rl_shared_policy_engine.py) | 12 | `a7365094c4df496d11aa797a9226cfc097d061b0f5ce136b3105603e29e6b145` |
| [fine_tuning/tests/test_notebook_rl_shared_smoke.py](../../fine_tuning/tests/test_notebook_rl_shared_smoke.py) | 10 | `19f841f7b3131ba5443dffa229e552da6eaa0f8b8a9948e436221bbf0fea4780` |
| [fine_tuning/tests/test_notebook_rl_smoke_data_evidence.py](../../fine_tuning/tests/test_notebook_rl_smoke_data_evidence.py) | 8 | `538a39f52acf31bf0927196e5097c04bd4652e84c164293e169e2bbd6d47c754` |
| [fine_tuning/tests/test_notebook_rl_source_observation.py](../../fine_tuning/tests/test_notebook_rl_source_observation.py) | 7 | `40f785fa92f278ec5652e6d04a7c44f1364ac8566146f283b247ca2b5b400ba0` |
| [fine_tuning/tests/test_notebook_rl_streaming_scores.py](../../fine_tuning/tests/test_notebook_rl_streaming_scores.py) | 9 | `9eed52cb5665c7dd71f480d1ebcb5b0a037689acbea706c73d4d99f7f57e9928` |
| [fine_tuning/tests/test_notebook_rl_strict_admission.py](../../fine_tuning/tests/test_notebook_rl_strict_admission.py) | 14 | `efe4ad3e7e6294213a39f9a8c4a5dd475cd6c3689e82e7016e766527aac17a51` |
| [fine_tuning/tests/test_notebook_rl_synthesis_credit.py](../../fine_tuning/tests/test_notebook_rl_synthesis_credit.py) | 33 | `086eb9dce5eeb6460c4efed6e822dbd7091d79f27ed66748bd8df70c7c36fbfb` |
| [fine_tuning/tests/test_notebook_rl_synthesis_credit_admission.py](../../fine_tuning/tests/test_notebook_rl_synthesis_credit_admission.py) | 16 | `768324ad4c8f0ec9126ae3660c180a75a03444a3eeb03bd67c0e4c1ccb9ae00f` |
| [fine_tuning/tests/test_notebook_rl_synthesis_credit_updates.py](../../fine_tuning/tests/test_notebook_rl_synthesis_credit_updates.py) | 15 | `0d1c79c8cb20a496dab70d299417cf44ec353f99f8af9fe0331674c5abfcd821` |
| [fine_tuning/tests/test_notebook_rl_synthesis_fact_reinspection.py](../../fine_tuning/tests/test_notebook_rl_synthesis_fact_reinspection.py) | 12 | `e342c3dcc5558b535524ae119b4865814d016bf890640ee01389636a65b67257` |
| [fine_tuning/tests/test_notebook_rl_synthesis_format_feedback.py](../../fine_tuning/tests/test_notebook_rl_synthesis_format_feedback.py) | 17 | `0f677850408957d32c7b947f6293a56b67b8b2e942a3e8846204719ec4a6329c` |
| [fine_tuning/tests/test_notebook_rl_synthesis_integration.py](../../fine_tuning/tests/test_notebook_rl_synthesis_integration.py) | 22 | `b0b2820f106d07e3ce6064d55858306fd4caf80a725f0881eb8193457f7b8dc3` |
| [fine_tuning/tests/test_notebook_rl_synthesis_output.py](../../fine_tuning/tests/test_notebook_rl_synthesis_output.py) | 14 | `8cecf84a176d0a9f19c6d3cacfd82673304c3107952cc607c6ed820522c58800` |
| [fine_tuning/tests/test_notebook_rl_synthesis_profile.py](../../fine_tuning/tests/test_notebook_rl_synthesis_profile.py) | 12 | `dd2b99b3f99ecc1c491f7d7b04d4d6459625e4ed23f03121ceb4bab1bdd8846f` |
| [fine_tuning/tests/test_notebook_rl_synthesis_relationship_input.py](../../fine_tuning/tests/test_notebook_rl_synthesis_relationship_input.py) | 11 | `4a48467dcb4cb068c12b0073930b07ceebc0a5c5bf7f0470114ecbbd85fc4bc5` |
| [fine_tuning/tests/test_notebook_rl_synthesis_repair_parent.py](../../fine_tuning/tests/test_notebook_rl_synthesis_repair_parent.py) | 20 | `e09bafeb402b904a7e511279cda0fb6939cbd8baa0697de680af1f5a9b6c6a07` |
| [fine_tuning/tests/test_notebook_rl_synthesis_unknowns.py](../../fine_tuning/tests/test_notebook_rl_synthesis_unknowns.py) | 24 | `dc55730c1b017d2695ccde4794697ce96a241c13e14a15e3c4cf0a93ebd80af5` |
| [fine_tuning/tests/test_notebook_rl_tabmwp.py](../../fine_tuning/tests/test_notebook_rl_tabmwp.py) | 13 | `c1bd67725ee2e1e08a7400e2d39074e60a586c51f160a7d12792d61cc5028fb4` |
| [fine_tuning/tests/test_notebook_rl_trace_evidence.py](../../fine_tuning/tests/test_notebook_rl_trace_evidence.py) | 6 | `855939209d6938aa6960e125f4213cf7080306269c856da2ea8de8e47224abda` |
| [fine_tuning/tests/test_notebook_rl_trace_service.py](../../fine_tuning/tests/test_notebook_rl_trace_service.py) | 21 | `b06cb09806b3380425612656b6f3f011e79dad02d49b5a315eee8e4c1acf7935` |
| [fine_tuning/tests/test_notebook_rl_trainer.py](../../fine_tuning/tests/test_notebook_rl_trainer.py) | 19 | `3524659674890542d1465e897becb252c26238cca64a568141f6aed5580a836c` |
| [fine_tuning/tests/test_notebook_rl_updates.py](../../fine_tuning/tests/test_notebook_rl_updates.py) | 23 | `d926f9eb03b721910e6f42a9b89a3171b30d8beb5e82201cdeb3f6ed73f9e05f` |
| [fine_tuning/tests/test_notebook_rl_vault.py](../../fine_tuning/tests/test_notebook_rl_vault.py) | 24 | `1707983f814c99034aafd9d260778ae601c99221f420d028881216d8e08d62bc` |
| [fine_tuning/tests/test_notebook_rl_verified_update_callers.py](../../fine_tuning/tests/test_notebook_rl_verified_update_callers.py) | 11 | `b2bdc683aa61ec28bf767540801f9b93e6af9203129c5c5e21b680966a18f04a` |
| [fine_tuning/tests/test_notebook_rl_verified_update_policy.py](../../fine_tuning/tests/test_notebook_rl_verified_update_policy.py) | 12 | `78e29d04feaba38d98577bff4af18cfdf3aabaae509c2ff3d63a6f7006e8071d` |
| [fine_tuning/tests/test_notebook_rl_verifier.py](../../fine_tuning/tests/test_notebook_rl_verifier.py) | 9 | `ba760835863d10c14519aebd001d57e68be9f9b51cdab4e1cf370653c03c1961` |
| [fine_tuning/tests/test_notebook_rl_verifier_service.py](../../fine_tuning/tests/test_notebook_rl_verifier_service.py) | 20 | `578c711b35ffa588f446e62f69df94b8c4cc2b2149315383fbe634fc77bc66b7` |
| [fine_tuning/tests/test_notebook_rl_whole_probe_integration.py](../../fine_tuning/tests/test_notebook_rl_whole_probe_integration.py) | 10 | `3eab1b1e737d626cae0c06137ad3feafe6573277558ba56ec6e5327039e52250` |
| [fine_tuning/tests/test_notebook_rl_whole_reference_credit.py](../../fine_tuning/tests/test_notebook_rl_whole_reference_credit.py) | 17 | `9bbdadb2a3cd94912a32984dddbabde14219a66bb3b9659e0a61a5e3992c726a` |
| [fine_tuning/tests/test_notebook_rl_whole_reference_notebook.py](../../fine_tuning/tests/test_notebook_rl_whole_reference_notebook.py) | 25 | `1f9a21e3239c166d8ffd6a0c613a993232ea399d75a032a91b7bf068762c9386` |
| [fine_tuning/tests/test_notebook_rl_whole_reference_pages.py](../../fine_tuning/tests/test_notebook_rl_whole_reference_pages.py) | 19 | `0b1ee94bf345e29218860a6fa55ce9844b18e2fd52bea63a508c9ec09de3093f` |
| [fine_tuning/tests/test_notebook_rl_whole_reference_probe.py](../../fine_tuning/tests/test_notebook_rl_whole_reference_probe.py) | 15 | `02ecd776f6336753863af9c0ba297882a687758f1e2c18dbcc61c73f97794822` |
| [fine_tuning/tests/test_notebook_rl_whole_reference_profile.py](../../fine_tuning/tests/test_notebook_rl_whole_reference_profile.py) | 8 | `28f430d1b2905ba3294e5814ec3f62a5c979341ab8229f81e2844fac78dfb617` |
| [fine_tuning/tests/test_notebook_rl_whole_reference_trained_probe.py](../../fine_tuning/tests/test_notebook_rl_whole_reference_trained_probe.py) | 7 | `42c3bf51a13398ca78791c5d8e48f3ce34d117a5a97dd8e227be15420b2db524` |
| [fine_tuning/tests/test_notebook_rl_whole_reference_training_credit.py](../../fine_tuning/tests/test_notebook_rl_whole_reference_training_credit.py) | 15 | `a206fe659bb34f1d040e32849e0be65400535fac80d669af6af6ef6ff746c662` |
| [fine_tuning/tests/test_notebook_rl_whole_reference_training_episode.py](../../fine_tuning/tests/test_notebook_rl_whole_reference_training_episode.py) | 19 | `74ee47b99ff34b3b590afa8762b9bc0e1b186e4234aa8554884605d43fe93135` |
| [fine_tuning/tests/test_notebook_rl_whole_synthesis_probe.py](../../fine_tuning/tests/test_notebook_rl_whole_synthesis_probe.py) | 13 | `f76ed2f2a0524697eddc047067cfd36e4dd355f7bbe28305b5730463e0f5e0a4` |
| [fine_tuning/tests/test_notebook_rl_whole_trained_probe.py](../../fine_tuning/tests/test_notebook_rl_whole_trained_probe.py) | 11 | `e0b3811e9df1b9826c41f1a3f97bbddbbb3b395582f381f59c09b22f55719d68` |
| [fine_tuning/tests/test_nvw.py](../../fine_tuning/tests/test_nvw.py) | 12 | `0644caf8c955d3d6760b934e0c1283e11231acab0a187c1447a3920a1b5623f6` |
| [fine_tuning/tests/test_prepare_controls.py](../../fine_tuning/tests/test_prepare_controls.py) | 4 | `3fe3b9aaca1db9fcf1ebca678e06dd0c65171309bc6c89f37409eb8e20e2bbd5` |
| [fine_tuning/tests/test_prepare_training.py](../../fine_tuning/tests/test_prepare_training.py) | 31 | `da8a0640c3aa25b4c88decc3d4930dead4cf15435dfdbacf784e1e78dc69192a` |
| [fine_tuning/tests/test_professional_transfer.py](../../fine_tuning/tests/test_professional_transfer.py) | 18 | `a770e2b8e7fb2f568d9900573f91ea5a12a522cc8bcd4330e65625e98dc214aa` |
| [fine_tuning/tests/test_recovery.py](../../fine_tuning/tests/test_recovery.py) | 8 | `a515569d1db52611a2864f07d92e1e61d4db9cd97df931812b5f11d8343244a2` |
| [fine_tuning/tests/test_report.py](../../fine_tuning/tests/test_report.py) | 5 | `99432801d6c3d20712b23ca9375ff0dad7f70ef45bc3b3e50e5be7fd075cd21e` |
| [fine_tuning/tests/test_runpod_g6.py](../../fine_tuning/tests/test_runpod_g6.py) | 3 | `1fe2d6b9d39b61bc9b5cfb38a63aa1415d47a9f5cc8cb3b4be2944da4de4a1f3` |
| [fine_tuning/tests/test_runpod_setup.py](../../fine_tuning/tests/test_runpod_setup.py) | 6 | `54f8e1cce8ca83f0e9e37a4f8d1cbb0436edd7fc89f424860bf1afe5e32cccf6` |
| [fine_tuning/tests/test_tabmwp.py](../../fine_tuning/tests/test_tabmwp.py) | 14 | `6f819ff441abeaec2a6703224b1875fd372a058640b98c70f4ce268dc79aa652` |
| [fine_tuning/tests/test_tabmwp_evaluate.py](../../fine_tuning/tests/test_tabmwp_evaluate.py) | 15 | `eee3bd24942d2187a8e9756740b82ac7e106f13e61c9d10f41203e0188ed83ca` |
| [fine_tuning/tests/test_tabmwp_images.py](../../fine_tuning/tests/test_tabmwp_images.py) | 7 | `aabaa1542866b460e64031bf4c16d0549b9f24b34d42e24ecb095836a1b20b2f` |
| [fine_tuning/tests/test_tabmwp_runpod.py](../../fine_tuning/tests/test_tabmwp_runpod.py) | 6 | `87ebe40e85391f1daa7670dd32ec27d505cd6b62c82ca2387820f4d748edf8b3` |
| [fine_tuning/tests/test_tabmwp_trial.py](../../fine_tuning/tests/test_tabmwp_trial.py) | 11 | `04b9ca0443b5087a5934e508ec6f478a76e6dcdc12fa3adfee9d34680a3b113f` |
| [fine_tuning/tests/test_training.py](../../fine_tuning/tests/test_training.py) | 19 | `3a5055a141ead76d7fd93125715d3531b73bbf1528a19a720c9d191ac5ef11f2` |


<a id="D19"></a>
## D19. 핵심 알고리즘 의사코드 — 아직 실행 코드 아님

```python
def run_problem(actor, private_verifier, engine, protocol):
    tree = generate_t1_and_record_every_member(actor, engine, protocol)
    for stage in (2, 3):
        fresh_audits = independently_audit_original_solutions(tree)
        frontier = latest_completed_stage(tree)  # T1→T2→T3, 과거 T1로 회귀 금지
        parent = select_by_frozen_rule(
            answer_failures_or_verified_process_contradictions(frontier, fresh_audits)
        )  # 기존 branch={1:1, 2:1}·원래 결정적 순서 유지
        if parent is None:
            break
        request = exact_parent_tokens_plus_fixed_feedback(parent)
        # same prompt, policy, images, different fixed seeds; n2 original members
        group = generate_joint_errorcheck_and_revised_solution(request)
        save_raw_before_grading(group)
        parse_and_grade_same_raw_spans(group)
        attach_independently_verified_repairs(tree, parent, group)
    close_solution_history(tree)
    if has_actual_answer_correct_candidate(tree):
        note_group = generate_all_history_bounded_notes(tree, n=4)
        save_raw_before_grading(note_group)
        audit_every_submitted_claim_and_report_uncovered_issues(note_group, tree)
    else:
        write_unresolved_record_without_fake_success_note(tree)
    tree.admission = existing_strict_routing(fresh_regrade(tree))
    return tree

def train_same_policy_batch(trees):
    assert_all_sources_tokens_images_and_policy_bound(trees)
    admit_only_resolved_routes(trees)
    credits = make_original_group_centered_component_vectors(trees)
    old_lp, ref_lp = compute_all_before_any_mutation(trees)
    # reference is frozen NF4 F0, not BF16 or another teacher model
    for original_action in every_preserved_action(trees):
        current_lp = forward_original_tokens_and_pixels(original_action)
        assert_current_old_alignment_before_update(current_lp, old_lp)
        keep = original_mis_policy(original_action, old_lp)
        accumulate_signed_and_positive_verified_corrected_vjp(current_lp, credits, keep)
        backward_answer_plus_units_plus_single_kl(current_lp, credits, keep)
    fresh_regrade_and_rehash_before_commit(trees)
    validate_finite_gradients_and_frozen_parameters()
    record_functional_and_repair_signal_separately()
    commit_at_most_one_optimizer_step_if_valid()
    save_actual_changes_and_synchronized_checkpoint()
```

이름은 설명용이며 현재 실행 가능한 함수 이름이 아니다. 구현은 D10의 기존 caller 안에 최소 분기로 넣는다. 실제 error path, budget, context, raw 저장, 예외 후 gradient 정리는 생략하면 안 된다.

<a id="D20"></a>
## D20. 최종 인수 기준

- [ ] 실제 pretrained checkpoint와 독립 NF4 저장묶음 확인.
- [ ] 추가 SFT/teacher/benchmark-derived 학습 경로 없음.
- [ ] joint ErrorCheck→RevisedSolution의 원래 토큰·이미지·부모 이력 보존.
- [ ] 성공 답에도 남은 과정 오류를 독립적으로 기록.
- [ ] 전체 시도를 본 최종노트의 거짓 주장·누락·정확한 수정 관계를 구분.
- [ ] gold/expected patch가 actor payload에 들어가지 않음.
- [ ] answer PG의 직접 토큰 계수가 ErrorCheck·DIAG에서0이며 공유 문맥·KL의 간접 영향과 구분.
- [ ] n2/n4 원래 그룹·고정 분모·MIS·cap상쇄·부호 포함 VJP 검사.
- [ ] 실제 nativeCORRECTED의 검증된 양성 부분·전체 signed 기여를 모두 확인하고 I/U/OUTPUT과 구분.
- [ ] 최종 답 전부 오답·최종노트 없음에도 검증된 online-only 수정 신호의 학습 진입 가능.
- [ ] 실제 업데이트·frozenbase·ON/OFF 각각의 변경/동결·새process복원 확인.
- [ ] 고정10·별도평가·노트보상제거 대조군의 범위와 실제 노출 보고.
- [ ] fine_tuning 단독 이식, 공유 BF16·MMDL·V8·역사 결과 보존.
- [ ] 기능 성공과 성능 효과를 분리하여 보고; 대규모는 별도 승인.

이 체크리스트는 설계 시점에는 미완료다. 후속 구현자가 실제 증거 파일·명령·출력·검사 범위를 연결한 항목만 완료로 바꾼다.

### 이번 문서 산출물의 검증 기록

2026-10-26, 코드 변경·모델 실행과 구분한 문서 검증 결과다.

| 검사 | 실제 확인 |
|---|---|
| 현재 소스 보존 | AST 조사 때 기록한 Python 363파일의 SHA256과 최종 실물이 모두 일치 |
| 구현 출발점 | 계획의 핵심 소스 pin12개·기존 시험 파일11개 존재 확인; 신규 시험 파일2개는 명시적으로 제안 상태 |
| 문서 연결 | 보고서 D00–D20, 계획 P00–P23, 음성 사례 N01–N30 존재·연결 확인 |
| HTML 정적 검사 | ID31개 중복0·로컬 링크53개 검증·외부 asset0·JavaScript `node --check` 통과 |
| 실제 브라우저 | 기존 Windows Chrome에서 23검사 통과: 검색·빈 결과·펼침·접기·인쇄 전후 상태 복원·키보드·내부 링크·오프라인 자원 |
| 화면 | iframe 폭1440/390에서 문서 client/scroll 폭1425/1425·375/375; 실제 데스크톱·모바일 흐름도 시각 확인 |
| 독립 검토 | FINALIZED·UNKNOWN schema·원래 n 중심화·양성/부호 합산 VJP·frontier·경로/F/S/E를 검토하고 모호한 요약 문구 수정 |
| 제출 저장소 | `MMDL/` clean, HEAD `197838c11557911a14a46d87b7f52fda0cac5b3c` 유지 |

최초 문서 검사기는 구현 예정 시험 파일까지 기존 파일로 요구해 실패했다. 기존11개와 신규2개를 명시 분리한 뒤 재검증했다. 이는 생산 코드 시험 실패나 기준 완화가 아니다. 이 표는 새 RL 코드 구현, CPU 학습 suite, GPU 학습, 8GB 적합성, 점수 개선의 통과 기록이 아니다.
