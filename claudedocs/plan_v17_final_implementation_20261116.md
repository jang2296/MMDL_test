# v17 최종판 구현 계획 — F0 시작, 단일 응답 RL, 자기 교정 SFT, 실행 안정화

- 작성일: 2026-11-16.
- 상태: **구현 계획**. v17 코드·자료 동결·학습·효과 검증은 아직 수행하지 않았다.
- 요청 범위: v16·v16b 코드와 회수 기록을 분석하고, 구현자가 사용할 계획서 한 파일 작성.
- 이번 작업에서 RunPod 조회·접속·변경, GPU 사용, 모델 다운로드·생성·학습·평가를 하지 않았다.
- 교수님 제출용 MMDL, 기존 V8 파서, 과제 baseline 558/900은 변경 대상이 아니다.
- 목표: 원본 Qwen에서 재현 가능한 RL 중심 자기개선 학습 후 V8 585/900 이상을 검증한다.
- 목표 점수 달성은 보장하지 않는다. 구현 완료·학습 신호 확보·성능 개선을 별도로 판정한다.
- 문서의 분석·예시는 학습 목표로 사용하지 않는다. 상위 모델이 작성한 풀이·교정을 학습시키지 않는다.
- v16b 회수 결과의 정본: [회수본 분석](analysis_v16b_recovered_20261116.md).
- 실행과 해시 원장: [PROJECT_STATUS](../docs/PROJECT_STATUS.md).
- 운영 보강: 사용자 지적에 따라 15시간 자동 차단을 폐기하고, 3,000문항 전체 학습과 14~17시간 내 평가·회수를 목표로 한다.
- 시간 목표 초과 예상은 보고·판단 사항이며 자동 축소·자동 차단 사유가 아니다. 무결성·학습 붕괴·실제 자금 보호는 별도 규칙이다.

## 0. 구현자가 먼저 읽을 결정

| 결정 | v17 기본안 |
|---|---|
| 시작점 | 지정 Qwen3-VL-4B-Instruct F0, 새 optimizer/scheduler/RNG |
| 학습 범위 | 원본 기저 동결, Vision LoRA·Merger 4개·LLM LoRA |
| 본학습 | 3,000문항, 16문항 단위 187묶음 + 마지막 8문항 |
| 준비 | 본학습과 별도 512개 후보, 수용된 자기 풀이·원 데이터 사람 풀이만 SFT |
| 기존 감시 | 200개를 개발용으로 유지, 학습 금지 |
| 생성 횟수 | 첫 풀이 1회, 관측 최대 1회, 노트 최대 1회, 교훈 재풀이 최대 1회 |
| 본체 | 첫 풀이 RL + 관측 RL |
| 보조 학습 | A1 정답 첫 풀이, C 정답 재풀이, N 검증된 교정 부분의 SFT |
| 주석 사용 | 비공개 검증기에만 사용. 정답 사실 목록을 모델에게 공개하지 않음 |
| 노트 | 실제 자기 주장 ID를 참조해 수정값 생성. 모델에게 문자 offset을 요구하지 않음 |
| 짧은 풀이 | 256토큰 하한 제거, 나머지 검사 전부 수행, 답만 있는 목표는 제외 |
| 반복 풀이 | 프로그램이 반복을 확인한 첫 풀이(LOOP_CONFIRMED)는 확인된 실패로 음의 PG(감지 구간, 확률 마스크 없음). 반복이 확인되지 않은 단순 잘림은 PG 제외 유지(§10.3.1) |
| RL 확률 | 실제 학습 생성 분포의 처리 후 전체 어휘 확률 |
| SFT 확률 | raw next-token softmax의 CE로 명시적으로 변경 |
| 정규화 | 본학습 각 채널의 고정 부모 분모 16, 없는 행에 가중치 재분배 없음 |
| 원본 보존 | 본학습은 응답 전반 최대 64위치 F0 KL, 준비 단계는 기존 첫 네 토큰 anchor 유지 |
| 생성 일정 | 기존 두 정책 버전·η=1 일정 유지 |
| 평가 | P0·B2 의미 조건·V8, 문항당 답 하나, 최종 모델 900문항 한 번, **합치지 않은 조정층(비병합) 서빙**(§15.4) |
| 비교 보고 | 감시200·효과 집합·MMMU900 모두 원본의 종료 상태(잘림·반복 / 정상 종료)별로 나눠 보고(§5.3) |
| 평가 동시성 | KV 검증된 8칸 연속 처리. 하나가 끝나면 다른 긴 응답을 기다리지 않고 다음 문항 투입 |
| 평가 재개 | 같은 등록·모델·seed·동시성의 미완료 ID만 재개, 완료 정답·오답·잘림은 모두 보존 |
| 시간 | 전체 실행 14~17시간 목표. 시간 상수로 본학습 차단·문항 축소하지 않음 |
| 실행 권한 | 이 문서 작성은 구현·게시·클라우드 실행 승인이 아님 |

숫자 가중치와 통과 문턱은 아래에서 **설계값**으로 표시한다.
과거 실측 최적값이라고 설명하지 않는다.
새 실행 전에 모든 설정과 판단 규칙을 봉인한다.
실행 도중 효과를 보고 손실·필터·자료를 변경하지 않는다.

## 1. 근거를 어떻게 해석했는가

### 1.1 확정한 관측과 미확정 원인

| 관측 | 확인 내용 | 확대해서 말하면 안 되는 것 |
|---|---|---|
| v15 | 같은 450문항에서 280, B2 275 | v15가 원본보다 확실히 우수함 |
| v16b MMMU | 별도 미완료 실행 59개와 9개 | 전체 900 성능이 원본보다 낮음 |
| 외부 감시 | F0 154 → 준비 155 → 최종 146/200 | RL만이 하락 원인임 |
| 실제 갱신 | 예정 슬롯 198, optimizer step 195 | 모든 문항에 교정 학습이 있었음 |
| 첫 풀이 | 정답 1,647, 직접 PG 제외 258 | 정답 풀이의 과정이 전부 정확함 |
| 교정 재풀이 | 정답 192, C 수용 16 | 제외된 정답 176개가 모두 좋은 SFT 목표임 |
| 길이 하한 | 정답 재풀이 136개가 256 미만으로 탈락 | 길이 조건만 없애면 136개 모두 수용 가능함 |
| 노트 | 1,096개 생성, N 2개 | 정확한 오류 교정 2개를 학습함 |
| 실제 N | 두 행 모두 error:null, 사실 5개 | 오류 원인과 수정 방법을 학습함 |
| 이미지 | v15·v16·v16b 모두 시각 학습 경로 존재 | 내부 Vision 단독 효과가 입증됨 |
| 음의 OBS | 후보 14,095토큰 중 194토큰만 유지 | 기울기 기여율이 토큰 비율과 같음 |

v17은 확인된 병목을 수정한다.
각 변경이 MMMU 점수를 몇 문항 올리는지는 아직 분리되지 않았다.
성능 주장은 외부 첫 풀이 비교와 완료된 MMMU 평가 뒤에만 한다.

### 1.2 운영 기록의 표현 정정

- GPU 평균 47%는 짧은 관측 창의 수치이며 전체 7시간의 평균으로 쓰지 않는다.
- CPU 한 코어가 바빴다는 사실만으로 모든 유휴 시간의 원인을 CPU로 확정하지 않는다.
- 회수 로그에서 KV 용량·동시성·완료 정체는 확인된다.
- 선점·재계산 반복은 강한 원인 후보이지만 선점 횟수 전부가 회수된 것은 아니다.
- 과목 순서 섞기는 긴 응답의 몰림을 줄일 수 있지만 KV 수용량을 보장하지 않는다.
- 설치 중 doctor 미실행은 검증 공백이다. 사후 학습 성공만으로 영향이 없었다고 일반화하지 않는다.
- v16b에는 이미 다음 묶음과 지연 노트·재풀이를 겹치는 일정이 있다.
- 단순히 “두 묶음을 겹치면 두 배 빨라진다”는 최적화 계획을 만들지 않는다.

## 2. 분석한 코드의 범위와 현재 연결

### 2.1 기준 소스

- 상위 Git HEAD: 0f10c338f366ca2568595a06fe275a0f46dd0cb7.
- MMDL_test HEAD: 20b21f6d64e1966bb17c565380c9598f520aa47e.
- 상위 작업공간은 dirty 상태다. HEAD만으로 분석한 소스를 식별할 수 없다.
- 구현 시작 시 실제 파일 SHA256 목록을 새 source manifest로 기록한다.
- 기존 실행용 동결본은 hotpatch하지 않는다.
- 기존 수정·미추적 파일을 reset하거나 덮어쓰지 않는다.

아래 표는 직접 또는 병렬 분석으로 확인한 관련 모듈이다.
핵심 경로의 함수 본문·호출·계약을 확인했고, 검사 파일은 관련 사례를 읽었다.
모든 과거 프로젝트 파일과 모든 테스트 본문을 전수 검토했다는 의미는 아니다.

| 현재 파일 | 역할 / 확인한 주요 함수 | v17 처리 |
|---|---|---|
| fine_tuning/error_note/protocol.py | plain_protocol_v16b, validate_protocol_v16b, v16b_contract_for_update | 새 버전 builder·validator·update contract 추가 |
| fine_tuning/error_note/campaign_v16b.py | prepare, run_campaign, require_complete, export_metadata | 버전별 도구 선택, A1·N·C·신호 관문 연결 |
| fine_tuning/error_note/v16b_runtime.py | V16bScheduler, deterministic_update, run_v16b | 일정 보존, schema/행 종류·진행 기록 확장 |
| fine_tuning/error_note/v16b_rows.py | native_note_contract, verify_note_v16b, c_row_with_receipt | 옛 의미 보존, 순수 helper만 재사용 |
| fine_tuning/error_note/train_v16b.py | encode_v16b_row, block validator, backward_v16b_block | 옛 경로 보존, 새 loss는 별도 모듈 |
| fine_tuning/error_note/prep_v16b.py | preflight_items, select, generate, rows | 버전별 수용 규칙·능력 목록·전수 탈락 기록 |
| fine_tuning/error_note/data_v16b.py | prepare, load_items, validate, _checked_inputs | 새 capability/holdout 계약 검증 |
| fine_tuning/error_note/curation_v16b.py | apply_family_supplement, validate_selection | 분리·오염·원문 무결성 helper 재사용 |
| fine_tuning/error_note/curation_policy_v16b.py | choose, interleave | 결정적 대체·순서 생성 재사용 |
| fine_tuning/error_note/curation_review_v16b.py | 출처 검토·검수 결과 연결 | 검수 범위와 AI 검토 표기 유지 |
| fine_tuning/error_note/guard_v16b.py | GuardV16B, funding_decision | v17 신호 관문과 기존 통계 감시 분리 |
| fine_tuning/error_note/content_v16.py | graph/geometry/scene/chart 연결, load_content_manifest | 비공개 검증 근거 연결 재사용 |
| fine_tuning/error_note/data_v16.py | admit_and_split | 과거 holdout 원장 읽기, 과거 분할 변경 금지 |
| fine_tuning/error_note/self_note_v16.py | public_note_catalog, verify_retry, first_solution_row | 검사 조각 재사용, 옛 전체 수용기 직접 사용 금지 |
| fine_tuning/error_note/train_v16.py | token_byte_mask, mergers_original | 토큰 소유권·F0 Merger 복원 재사용 |
| fine_tuning/error_note/run_v16.py | 실행 준비·reload·effect 검사 | RNG/helper 재사용, O-only 효과 계약은 재사용 금지 |
| fine_tuning/error_note/single.py | run_joint_single_update, save/load continuation | v17 dispatch·행 소비·slot 갱신 |
| fine_tuning/error_note/single_campaign.py | 옛 v15/v16 캠페인 | 보존, 실행 이력과 비교 근거만 읽기 |
| fine_tuning/error_note/single_grading.py | grade_single | 기존 학습 채점 의미 유지 |
| fine_tuning/error_note/train_v4.py | encode_single_row, backward_single_block, raw KL helper | v17 dispatch와 계산 helper 재사용 |
| fine_tuning/error_note/rl.py | processed_full_logprobs, negative_token_mask | 실제 sampling 확률 재사용 |
| fine_tuning/error_note/single_vllm.py | submit, step, prefill_barrier, add_version, _pf5 | telemetry와 검정 단계 명시 |
| fine_tuning/error_note/llm_lora.py | 정책 전송·시각 hook·slope_check | 비병합 서빙·검사 재사용 |
| fine_tuning/error_note/lora_preflight.py | engine_checks | F0 검정력 없음과 적용 실패 구분 |
| fine_tuning/error_note/plan.py | _plan_v16b, plan | v17 단계·doctor·평가 용량 gate 추가 |
| fine_tuning/notebook_rl/proof_eval.py | registration, run_eval, export, paired 분석 | v17 실행 변형 추가, 보호된 src 수정 없이 연결 |
| fine_tuning/notebook_rl/pod/ | runner, start, collect, common, data bundle | 감시·수집·복구 helper 재사용 |

### 2.2 현재 주요 호출 사슬

    plan._plan_v16b
      → prep_v16b.select / generate / rows
      → campaign_v16b.prepare
      → campaign_v16b.run_campaign
          → V16bScheduler
          → request / complete / build_rows callbacks
          → train_v4.encode_single_row
          → single.run_joint_single_update
              → train_v4.backward_single_block
              → train_v16b.backward_v16b_block
          → checkpoint + drain state + fresh reload
      → proof_eval export / preregistration / run_eval
      → V8 analyze / separate Compass / recovery

행 생성만 고치면 v17 구현이 끝나지 않는다.
encoder, block validator, 손실 dispatch, 소비 원장, 재개, 내보내기까지 같은 버전 계약으로 연결해야 한다.

### 2.3 실제로 수정이 닿아야 할 지점

| 지점 | 현재 동작 | 필요한 변경 |
|---|---|---|
| campaign run_campaign의 request | v16b catalog를 NOTE 입력에 전달 | v17은 모델 생성 claim ledger만 전달 |
| complete | T1/OBS/NOTE/RESOLVE별 receipt | 관측 주장 감사·교정·교훈·C를 별도 receipt로 연결 |
| build_rows | T1에서 RL만 생성 | 같은 T1으로 RL과 조건부 A1 생성 |
| encode_v16b_row | N만 span mask 재계산 | v17 encoder에서 kind별 마스크를 명시적으로 재검증 |
| _validate_v16b_block | 정해진 kind·phase·schedule 검사 | A1의 부모·중복 소비·P0 입력 검사 추가 |
| run_joint_single_update | v16b 버전만 prep/flush 특별 처리 | v17도 같은 슬롯·HOLD·소비 불변식 적용 |
| run_v16b | 고정 지연 일정 | 새 행을 올바른 기존 슬롯에만 넣기 |
| _plan_check | v16b 버전·동시 32 고정 | v17 계약은 해당 protocol에서 재구성해 검증 |
| proof_eval 버전 목록 | 기존 모델·학습 버전 제한 | v17 export·preregistration·variant 허용 |
| proof_eval 등록·run_eval | START1회, backend 생성 callback에서 등록 검사 | 동일 START 재개·완료 shortcut 전 검사·생성 후 KV receipt |
| pod.chain / collect | 기존 chain 재실행 거절, 분석용 pack_eval | 평가 전용 재개 CLI·재개에 필요한 samples/raw/identity 회수 |
| GuardV16B / funding_decision | 급격한 short 붕괴·고정 평가 예약 | v17 추세 경고·실측 평가 예약, 옛 계약 보존 |

## 3. 유지할 규칙과 새 버전 경계

### 3.1 학습·평가 규칙

1. 상위 모델의 풀이·노트·정답 설명을 학습시키지 않는다.
2. 첫 풀이에는 정답·노트·검증 결과·참고 풀이를 넣지 않는다.
3. 정답은 채점 뒤 NOTE 입력에서만 모델에게 제공한다.
4. 데이터셋이 제공한 사람 작성 풀이의 준비 SFT는 허용 범위에서 사용한다.
5. 자유 자연어 설명의 정확성을 AI가 승인했다는 이유로 VERIFIED로 바꾸지 않는다.
6. MMMU/MMMU-Pro 모든 split과 파생물은 학습·증강에서 제외한다.
7. 평가는 원래 P0와 단일 답이다. OBS/NOTE 추가 프롬프트를 평가에 넣지 않는다.
8. 그룹 4회 풀이·다수결·best-of-N은 추가하지 않는다.
9. 학습 모델 선택은 사용자가 허용했지만 문항별 답 선택과 섞지 않는다.
10. 과거 점수·필터·프로토콜을 v17 기준으로 소급 덮어쓰지 않는다.

### 3.2 코드 변경 범위

- 새 실험 구현은 fine_tuning 아래에 둔다.
- 기존 V8 파서와 src/mmdl의 기존 구현은 변경하지 않는다.
- 평가 조정은 기존 proof_eval의 등록·backend 어댑터 경로를 사용한다.
- 교수님 MMDL checkout은 변경하지 않는다.
- 과거 hash 고정 modules.py, train.py, protocol.json, notebook_rl의 guards/vault/schema는 수정하지 않는다.
- 여기서 train.py의 순수 계산 helper를 호출할 수는 있으나 파일을 수정하지 않는다.
- 공유 모델 cache·원본 가중치·다른 프로젝트 자원은 변경하지 않는다.
- 새 기능은 v17 버전에서만 활성화한다.
- 기존 버전의 순수 builder 출력·fixture·프로토콜 hash를 회귀 검사한다.
- 소스 파일이 바뀌면 새 source manifest가 필요하다. 과거 동결 manifest를 새 소스에 맞게 고치지 않는다.

### 3.3 최소 모듈 구성

새 파일은 기능 경계가 필요한 네 개를 기본으로 한다.
기존 캠페인·스케줄러·체크포인트 구현을 통째로 복제하지 않는다.

| 새 파일 | 책임 |
|---|---|
| fine_tuning/error_note/v17_rows.py | OBS claim, NOTE repair, A1/C/N 수용·마스크·receipt |
| fine_tuning/error_note/train_v17.py | v17 encoder·검증·RL/CE/KL 손실 |
| fine_tuning/error_note/v17_data.py | 과거 holdout·노출 감사, capability manifest, v17 동결 입력 CLI |
| fine_tuning/error_note/v17_eval.py | KV 사전 검사·heartbeat·v17 평가 계약 보조 |

protocol.py에는 새 상수·builder·validator를 추가한다.
campaign_v16b.py에는 명시적인 v17 선택 분기를 추가한다.
파일 이름이 v16b라는 이유로 과거 실행 계약을 v17로 바꾸지는 않는다.
CLI의 기본값은 기존 v16b이며 v17은 명시적으로 선택한다.

공통화는 실제로 같은 동작을 두 번 작성해야 할 때만 한다.
범용 plugin registry나 새 학습 framework는 만들지 않는다.
private helper 재사용 시 인자·예외·상태 복구 조건을 테스트에 묶는다.

## 4. 데이터 결정 — 재수집보다 권한·연결·능력 범위부터

### 4.1 기본 수량

| 역할 | 부모 문항 수 | 실제 학습 여부 |
|---|---:|---|
| main | 3,000 | 한 번씩 T1 일정 소비, 유효 신호만 학습 |
| prep | 512 | 통과한 SELF/HUMAN 행만 학습 |
| development | 200 | 가중치 학습 금지 |
| 합계 | 3,712 | 학습 부모 최대 3,512 |

외부 효과 확인용 holdout은 이 3,712개와 별도로 동결한다.
이를 3,712개 학습이라고 보고하지 않는다.
원 데이터의 source_split=train과 우리 역할 main/prep/development를 구별한다.

### 4.2 현재 3,000개 구성은 출발 자료이지 성능 최적 비율이 아니다

| 출처 | 현재 수 |
|---|---:|
| AI2D | 183 |
| ChartQA | 225 |
| CheBI | 396 |
| CLEVR | 48 |
| DeepWeeds | 69 |
| Geometry3K | 179 |
| GVLQA | 319 |
| MapQA | 450 |
| NCTCRC | 25 |
| PlantVillage | 26 |
| PTBXL | 222 |
| ScienceQA | 179 |
| SLAKE | 210 |
| SSMR | 449 |
| TQA | 20 |
| 합계 | 3,000 |

이번 계획은 이 비율이 MMMU에 최적이라고 주장하지 않는다.
출처 종류가 많다는 사실을 교정 가능한 내용의 다양성과 동일시하지 않는다.
학습 결과를 본 뒤 임의로 쉬운 문항을 교체하지 않는다.

### 4.3 실제 능력 범위는 별도로 센다

현재 코드와 조립 자료의 CPU 재집계 결과:

| 속성 | 현재 main 부모 수 |
|---|---:|
| 주석 참조 존재 | 668 |
| 과거 metadata의 annotation_verifier_supported | 192 |
| 실제 OBS 요청 적격 | 191 |
| 실제 native note catalog 지원 | 188 |

- OBS: 도형 47, 그래프 96, 장면 48.
- note catalog: 도형 44, 그래프 96, 장면 48.
- catalog 사실 수: 1,673.
- OBS 적격 도형 중 세 문항은 원자적 NOTE catalog가 없다.
- prep 512와 기존 development 200은 현재 OBS·native note catalog 지원이 0이다.

191개와 188개는 서로 다른 수치다.
capability를 boolean 하나로 합치지 않는다.
새 claim 계약에서는 이 수치도 다시 검증한다.

### 4.4 새로 발견한 과거 holdout 경계 문제

현재 v16b는 최신 v16 frozen-v4의 예약 자료를 기준으로 제외했다.
그러나 v16의 과거 네 동결 manifest를 합치면 nontraining ID가 530개다.
최신 v4의 sentinel 16 + dev 162 + sealed 320 = 498개보다 많다.

그 역사적 합집합 중 32개 ID가 현재 main에 있다.
구성은 Geometry3K 16개와 CLEVR 16개다.
이것만으로 과거 실행의 규칙 위반을 확정하지 않는다.
이전 분할을 해제하거나 재분할하도록 허용한 기록이 있는지 먼저 확인해야 한다.

v17의 보수적 기본 처리:

1. 과거 모든 예약 manifest와 결정 원장을 읽는다.
2. 명시적으로 해제된 reservation만 승인 기록 hash와 함께 제외 목록에서 푼다.
3. 해제 근거가 없거나 모호하면 예약을 유지한다.
4. ID뿐 아니라 native ID·family·이미지 byte/pixel identity까지 닫힌 집합으로 전파한다.
5. 현재 main의 해당 부모는 v17에서 격리한다.
6. 기존 적격 미사용 후보에서 결정적으로 대체해 main 3,000개를 채운다.
7. 기존 v16b manifest와 과거 결과는 변경하지 않는다.

32개를 그대로 격리하는 경우 정확한 ID 수준에서만:

- OBS는 191 → 159.
- native note catalog는 188 → 157.
- 추가 family/image 전파 뒤에는 더 줄 수 있다.

따라서 v17에 191개 관측 문항이 확보됐다고 미리 쓰지 않는다.
교정 문항 수를 채우려고 예약 자료를 학습에 돌리지 않는다.

### 4.5 기존 미사용 풀의 실제 한계

현재 적격 후보 5,468개에서 기존 3,712개 ID를 뺀 후보는 1,756개다.

| 출처 | 미사용 후보 ID |
|---|---:|
| ScienceQA | 577 |
| NCTCRC | 385 |
| ChartQA | 292 |
| MapQA | 238 |
| PlantVillage | 117 |
| DeepWeeds | 81 |
| TQA | 56 |
| ECG | 10 |

이는 추가적인 family/image 전파 전의 후보 수다.
현재 검증기가 지원하는 추가 OBS 후보는 0이다.
단순 재선별로 관측 학습량을 늘릴 수 있다는 이전 일반론을 이 실측에 맞춰 제한한다.

기존 후보를 활용하려면 새 원 주석 연결·검증 adapter가 먼저 필요하다.
새 주석이 존재하지 않으면 모델이나 분석 에이전트가 대신 만들어 넣지 않는다.
지원 자료가 부족하면 부족한 수와 이유를 그대로 보고한다.

### 4.6 재선별 알고리즘

입력:

- 기존 main/prep/development manifest.
- 모든 역사적 reservation manifest.
- 기존 적격 후보와 오염·품질·권한 receipt.
- 새 annotation adapter가 실제 검증한 capability inventory.

절차:

1. 현재 main 중 새 권한·분리 검사에 통과한 부모를 우선 유지한다.
2. 제거 사유는 reservation, family overlap, invalid annotation binding 등으로 정확히 기록한다.
3. 미사용 후보에 같은 hard eligibility를 적용한다.
4. 과거 출처 상한과 분리 규칙을 유지한다.
5. 필요한 대체 수만큼 검증 능력·원 사람 풀이 존재 여부·고정 hash 순으로 선택한다.
6. 정렬 키와 seed를 코드 상수·protocol로 고정한다.
7. 입력 파일 순서를 뒤집어도 선택 결과가 같아야 한다.
8. 3,000개를 채우지 못하면 DATA_INSUFFICIENT로 종료한다.
9. MMMU 정오·과목별 손실·기존 monitor 실패 문항과의 유사도를 선택 점수에 사용하지 않는다.

구체적 정렬 키의 설계값:

    (-source_bound_repair_capable,
     -native_human_reference_present,
     sha256("v17-replace:3407:" + problem_id),
     problem_id)

출처 상한은 먼저 적용한다.
이 정렬은 새 생성 결과를 사용하지 않는다.
선택에서 빠진 자료를 실패 응답을 보고 다시 끼워 넣지 않는다.

### 4.7 capability manifest

부모마다 다음 필드를 봉인한다.

    schema: v17-capability-v1
    problem_id
    assembled_role
    source_id / source_revision / source_split / native_id
    family_ids
    image_byte_sha256 / image_pixel_sha256
    training_allowed
    answer_verifier_id
    annotation_present
    annotation_source_refs
    annotation_binding_receipt_sha256
    obs_eligible / obs_reason
    repair_catalog_eligible / repair_reason
    supported_claim_kinds
    closed_world_scopes
    native_human_reference_refs
    public_schema_sha256
    private_verifier_sha256
    verifier_source_sha256
    inherited_reservations

manifest는 선언을 신뢰하는 허가증이 아니다.
load 단계에서 실제 원 주석과 검증 코드로 다시 계산하고 일치 여부를 확인한다.
공개 스키마 hash와 비공개 사실 hash를 분리한다.
비공개 자료 객체 전체를 prompt에 직렬화하지 않는다.

### 4.8 준비 512의 편향을 숨기지 않는다

현재 prep은 MapQA 232, ChartQA 116, ScienceQA 92, DeepWeeds 36,
NCTCRC 13, PlantVillage 13, TQA 10이다.
지도·차트 합계 348/512, 약 68.0%다.
main의 여덟 출처는 prep에 없다.

v17 기본안은 별도 prep 512의 ID를 유지한다.
여기에 없는 영역의 F0 능력이나 OBS baseline을 측정했다고 보고하지 않는다.
prep 수용 규칙 변경 뒤 실제 SELF/HUMAN 행 수·출처·길이를 새로 기록한다.
과거의 25 SELF + 77 HUMAN을 v17 수량으로 고정하지 않는다.

준비 데이터를 균형 있게 다시 선정하려면 별도 v17 manifest를 만들어야 한다.
그것은 기본 실행 도중 자동으로 하는 변경이 아니다.

### 4.9 원 사람 풀이의 출처

- 원 데이터가 제공한 풀이 필드만 사용한다.
- 답 라벨의 origin이 HUMAN이라는 이유로 설명도 사람이 썼다고 추정하지 않는다.
- ScienceQA lecture/solution은 원문과 출처·필드·hash를 연결한다.
- 허용된 텍스트의 오염 검사 receipt를 유지한다.
- 사람이 만든 데이터라는 출처와 우리 팀이 사람이 검수했다는 인증은 구별한다.
- AI 검토는 human_verified=false다.
- 문서 작성 에이전트가 만든 설명을 사람 풀이로 넣지 않는다.

## 5. 새 문항 순서와 효과 확인 자료

### 5.1 본학습 순서

기존 interleave를 재사용한다.
새 최종 목록에서 source, 실제 obs/repair capability, 허용된 학습 자료의 F0 prior를 층으로 사용한다.
모든 층의 누적 편차를 계산해 receipt에 남긴다.
기존 실행의 최대 편차 수치를 새 순서에 그대로 복사하지 않는다.

- 묶음 0..186: 16개.
- 묶음 187: 8개.
- 미수용 부모를 새 부모로 대체하지 않는다.
- 각 부모의 T1은 한 번만 생성한다.
- 생성 완료 순서가 학습 부모 순서를 바꾸지 않는다.

### 5.2 외부 자료의 역할

| 집합 | 용도 | 금지 |
|---|---|---|
| 기존 200 | 개발 감시·명시적 모델 선택 보조 | 독립 확증 집합이라고 부르기 |
| 기존 v16 dev/sealed/sentinel | 예약·노출 감사 후 적합한 평가 용도 검토 | 이름만 보고 미학습으로 인정 |
| 새 효과 manifest | F0와 선정 v17의 첫 풀이 비교 | 결과를 보고 문항 교체 |
| MMMU 900 | 최종 기준 평가 | 학습·개별 개선 자료로 사용 |

v15 추가 실행 256개만 보아도 기존 v16 dev 20개, sealed 40개, sentinel 2개와 ID가 겹친다.
현재 development 200에도 v15 추가 실행과 겹치는 ID 3개가 있다.
이것은 v17 F0 시작의 학습 오염을 뜻하지 않지만,
해당 집합으로 v15와 v17의 미학습 일반화를 공정하게 비교한다는 주장은 제한한다.

모델 비교에 사용하려면 각 모델의 **실제 실행된 조상 학습·재생 이력**을 합친다.
단순 캠페인 계획 포함과 실제 학습 노출을 구별한다.
그 이력을 native/family/image identity로 전파한다.

### 5.3 효과 판정 계약

- 학습 전에 효과 집합 ID·질문·이미지·seed·sampling·채점 버전을 고정한다.
- F0와 최종 모델은 같은 집합을 P0 첫 풀이 한 번으로 푼다.
- 새 효과 집합의 설계 상한은 128개, 최소 적격 공급은 64개다.
- 기존 미사용 적격 후보에서 v17 main/prep/development와 family/image 겹침을 제외한다.
- 이미 결과를 분석·설계·모델 선정에 사용한 평가 자료와 그 파생물도 제외한다.
- 역사적 reservation은 자동 해제하지 않는다. 이를 새 효과 집합에 편입하려면 원래 용도와 노출 이력을 확인한다.
- 출처별 남은 후보 비율에 largest-remainder 할당을 적용하고 source 이름으로 동점을 푼다.
- 출처 안에서는 `sha256("v17-effect:3407:" + problem_id)` 순으로 고른다.
- 후보가 64..127개이면 전부, 128개 이상이면 위 규칙의 128개를 고정한다.
- 기존 F0 정오·v16b 정오·길이·좋은 결과를 selection key로 사용하지 않는다.
- 64개 미만이면 EFFECT_DATA_INSUFFICIENT로 표시하고 독립 효과 gate를 통과한 것으로 보고하지 않는다.
- 임의의 작은 표본을 MMMU 대표 표본이라고 하지 않는다.
- 회복·손실 수, paired 차이, 정확한 McNemar 또는 사전 지정 paired CI를 보고한다.
- 5% 유의성 미달을 동일 성능의 증거로 쓰지 않는다.
- “순이득 몇 개면 항상 유의하다” 같은 고정 개수 규칙을 사용하지 않는다.
- 기존 v16 O-only effect 계약은 PG=0을 요구하므로 그대로 통과시키지 않는다.
- exact_paired와 원문 검증 helper만 재사용하고 v17 evidence schema를 새로 만든다.

출처별 분리와 결과 미사용 이력을 증명하지 못하면 명칭은 외부 개발 비교다.
이미 본 자료를 새 파일로 옮겼다는 이유로 독립 확증 집합이라고 하지 않는다.
v15까지 비교할 경우에는 §5.2의 조상 학습 노출 합집합을 추가 제외한다.

### 5.4 원본 종료 상태별 분해 보고

v11b~v15는 모두 "끝맺기만 좋아지고 정상 종료 문항의 정답률은 그대로"였다.
v15의 같은 450문항도 원본이 잘렸던 46문항은 1→9, 정상 종료 404문항은 274→271이었다.
v17이 내용 정답률을 실제로 올렸는지 구별하려면 모든 비교를 원본 기준 종료 상태로 나눠야 한다.

| 비교 | 기준 응답 | 층 |
|---|---|---|
| 개발용 200 | 같은 문항의 F0 응답 | F0 잘림·반복 / F0 정상 종료 |
| 효과 집합 | 같은 문항의 F0 응답 | 같음 |
| MMMU 900 | B2 응답(고정 기준) | B2 32K 도달 90문항 / B2 정상 종료 810문항 |

- 층마다 기준 정답 수, v17 정답 수, 회복·손실, 정확한 McNemar p를 보고한다.
- 정상 종료 층은 다시 "둘 다 정상 종료"와 "v17만 잘림·반복"으로 나눈다(v15 분석의 398문항 271→271 같은 비교).
- 층 기준은 기준 응답의 finish_reason과 `is_loop` 결과로 실행 전에 정해지며, v17 결과를 보고 층을 바꾸지 않는다.
- 전체 순변화가 양수여도 정상 종료 층이 0 이하이면 "내용 정답률 개선 미입증"으로 쓴다.

## 6. 생성 계약과 입력 경계

### 6.1 공통 생성 설정

본학습 T1·OBS·NOTE·RESOLVE sampling:

    temperature: 0.7
    top_p: 1.0
    top_k: 0
    presence_penalty: 1.5
    repetition_penalty: 1.0
    num_beams: 1

PREP_SELF 수집과 외부 감시·평가 sampling은 기존 B2의 top_p 0.8, top_k 20을 유지한다.
준비 OBS가 실제로 존재한다면 위 본학습 sampling으로 수집하며, 현 prep에는 적격 OBS가 없다.
PREP_SELF 생성 receipt의 processed-truncated-b2 표기는 보존하고, 학습 CE 공간과 별도로 검사한다.
두 분포가 다름을 기록하고 같은 분포라고 주장하지 않는다.
자동 temperature/정밀도/해상도/토큰 축소는 없다.

기본 상한:

| 단계 | 상한 |
|---|---:|
| T1 | 8,192 |
| OBS | 1,024 |
| NOTE | 1,024 |
| RESOLVE | 8,192 |
| PREP | 8,192 |
| 외부 개발 감시 | 8,192 |
| MMMU 최종 평가 | 32,768 |

NOTE prompt의 현재 16,384 입력 상한을 유지한다.
초과하면 원문을 몰래 자르지 않고 NOTE_CONTEXT_BLOCKED로 기록한다.
새로운 압축 프롬프트를 자동 fallback으로 넣지 않는다.

### 6.2 단계별 허용 정보

| 정보 | T1 | OBS | NOTE | RESOLVE | 평가 |
|---|---|---|---|---|---|
| 질문·보기 | 원래 P0 | 없음 | 있음 | 원래 P0 | 원래 P0 |
| 원본 이미지 | 있음 | 있음 | 있음 | 있음 | 있음 |
| 정답 | 없음 | 없음 | 채점 후 허용 | 없음 | 없음 |
| 원 주석 정답값 | 없음 | 없음 | 없음 | 없음 | 없음 |
| 실제 자기 주장 | 없음 | 자기 생성 | 있음 | 교훈 외 없음 | 없음 |
| 검증 결과 | 없음 | 생성 후만 | 범위 표시한 결과 | 없음 | 없음 |
| 누설 검사한 교훈 | 없음 | 없음 | 자기 생성 | 있음 | 없음 |

OBS는 별도의 학습 보조 과제다.
OBS에서 맞힌 사실을 최종 평가 답으로 선택하지 않는다.
OBS 오류를 T1 오답의 내부 원인이라고 자동 인증하지 않는다.

## 7. OBS — 원 주석은 숨기고 모델의 실제 주장을 검증

### 7.1 모델 출력

출력 형식은 소스별 일반 schema만 공개한다.
실제 정답 entity 목록·정답 간선 목록·객체 개수를 prompt에 넣지 않는다.

    {
      "claims": [
        {
          "kind": "SEGMENT_LABEL",
          "entity": ["A", "B"],
          "value": "5"
        }
      ]
    }

위 예시는 계약 설명용이며 실제 학습 자료로 사용하지 않는다.
모델이 entity와 value를 이미지에서 직접 작성한다.
claim ID는 모델 출력 필드가 아니다.

### 7.2 최초 지원 범위

| 출처 | claim 종류 | 검증 경계 |
|---|---|---|
| Geometry3K | SEGMENT_LABEL, ANGLE_LABEL, PARALLEL, PERPENDICULAR | diagram logic에 직접 있는 사실 |
| GVLQA | VISIBLE_NODE, EDGE_PRESENT, EDGE_WEIGHT | 이미지와 연결된 완전한 구조의 인증 범위 |
| CLEVR | OBJECT_ATTRIBUTE, OBJECT_COUNT | scene identity·보이는 객체 정의가 일치하는 범위 |
| ChartQA | CHART_VALUE | 원표·series/category·단위·가시성 연결이 인증된 항목만 |
| AI2D·MapQA·기타 | 초기에는 UNSUPPORTED 또는 명시된 부분 범위 | QA 라벨만으로 시각 사실을 인증하지 않음 |

새 adapter가 없는 출처는 답 RL에서 계속 사용할 수 있다.
원 주석 참조가 있다는 이유만으로 OBS와 N을 허용하지 않는다.
차트 adapter 구현은 필수 연결 작업이지만, 수용 문항 증가 자체는 보장하지 않는다.
W1~W2의 선행 필수 작업으로 두고, ChartQA 225개 본학습 후보의 실제 연결·가시성·순서 인증부터 점검한다.
content_v16.chart_target의 원표 join·숫자·단위 검사는 재사용하되, 반환된 설명 문장은 학습 목표로 쓰지 않는다.
이 함수의 OCR 숫자 존재 확인만으로 막대/series/category 위치까지 인증하지 않는다.
기존 코드에서 PEW_NO_SOURCE_TABLE, UNSUPPORTED_CHART, VISIBILITY_UNVERIFIED인 자료를 자동 수용하지 않는다.

차트 연결 뒤 본학습 동결·유료 실행 판단 전에 아래 수량을 source별로 보고한다.

    후보 수 → 원표/주석 있음 → 이미지 join 확인 → 위치·단위·가시성 확인
           → 고정 관측 scope 확인 → 출력 크기 통과 → OBS 적격 / repair 적격

서로 다른 수치를 단일 annotation_supported로 합치지 않는다.
현재 191→159는 과거 reservation의 ID 제외만 적용한 수이며 새 scope 검사 뒤에는 더 줄 수 있다.
차트가 늘지 않으면 주요 관측 교정 경로가 본학습의 약 5.3% 이하라는 한계를 실행 판단에 명시한다.
나머지 문항은 주로 첫 풀이 RL+A1이고, 검증 가능한 산술·재풀이 경로가 추가될 수 있다.
이는 94%가 모든 교정 방법에서 영구 제외된다는 뜻은 아니다.
미사용 ChartQA 292개도 같은 원자료 인증을 거쳐야 하며, 기존 규칙 안의 재선별만 허용한다.
새 adapter가 완성됐다는 이유만으로 225개 전체 관측 학습이나 65점 개선을 예상하지 않는다.

### 7.2.1 출력 크기를 먼저 제한하는 고정 관측 범위

1,024토큰 안에 표현할 수 없는 전수 목록을 요구한 뒤 잘림으로 버리는 일을 막는다.
관측 범위는 정답·실제 응답·모델 성공 여부와 무관한 다음 규칙으로 고정한다.

| 출처 | 공개 지시의 범위 | 비공개 target F의 동일 범위 |
|---|---|---|
| 그래프 | 보이는 숫자 노드 ID 오름차순 앞 4개와 그 노드 사이 간선 | 인증된 visible node 앞 4개, 해당 induced subgraph |
| 도형 | 직접 표시된 사실을 공개 kind 순서·entity 이름 순서로 정렬한 앞 8개 | 같은 정렬의 diagram fact 최대 8개 |
| 장면 | 왼쪽부터 앞 3물체의 크기·색·재질·모양과 전체 개수 | 동일 identity의 속성 12개 이하 + 개수 1개 |
| 차트 | 범례 순서의 series, 각 series의 가로 category 순서에서 앞 8개 표시값 | 가시성과 순서가 인증된 값 최대 8개 |

그래프 가중치는 별도 중복 claim 대신 해당 EDGE_WEIGHT의 값에 포함한다.
자기고리가 없는 단순 방향 그래프는 최대 4노드 + 12간선으로 max_claims=16 안에 들어간다.
자기고리·중복간선 등을 포함해 16claim을 넘으면 사전 제외하며 존재하는 간선을 숨겨 맞추지 않는다.
숫자 ID 순서·도형 entity 정렬·차트 범례/category 순서가 인증되지 않으면 그 범위는 UNSUPPORTED다.
정답 entity 목록·정답 개수·실제 values는 공개 prompt에 넣지 않는다.
문항마다 답을 보고 유리한 scope를 고르지 않는다.
공개 범위의 순서는 private annotation의 파일 순서가 아니라 그림에서 관찰 가능한 순서다.
도형 주석이 불완전해 보이는 앞 8개 fact와 주석 앞 8개 fact의 동일성을 인증할 수 없으면 OBS 제외다.
미주석 참 사실이 앞에 들어와 모델의 뒤쪽 출력이 달라진 것을 FN으로 감점하지 않는다.
ChartQA도 CSV 행 순서와 실제 범례·가로축 순서의 일치를 별도로 인증한다.

CPU capability 생성에서 고정 tokenizer로 canonical target JSON의 token 수를 측정한다.
설계 상한은 768token이며 1,024 생성 한도 안에 형식 여유를 둔다.
16claim/768token을 넘으면 이 계약의 OBS 적격에서 제외하고 reason을 기록한다.
이 canonical JSON은 용량·검증용 private 표현이며 모델 프롬프트나 SFT 목표로 전달하지 않는다.
실제 Qwen 출력이 더 길 수 있으므로 이 검사는 잘림 방지 보장이 아니다.
주석이 일부만 있는 도형은 범위 자체가 부분 인증임을 유지하고 다른 참 사실을 FP로 만들지 않는다.
새 범위로 OBS/N 수를 다시 계산한다. 기존 191/188/159/157을 확정 공급으로 재사용하지 않는다.

### 7.3 source-bound 검증

그래프:

- parse_gvlqa_structure의 구조·범위·중복·간선 수 검사를 재사용한다.
- 방향 그래프와 무방향 그래프를 혼합하지 않는다.
- 원 그래프·그림 hash·rendering 검토 receipt를 함께 고정한다.
- 보이지 않는 고립 노드는 보이는 노드 정답으로 넣지 않는다.
- closed-world가 인증된 범위에서만 edge 부재를 거짓/참으로 판정한다.

도형:

- link_geometry3k의 이미지·문장·답 연결을 재사용한다.
- 행 번호만으로 다른 문제의 주석을 붙이지 않는다.
- text logic이나 정답에서 그림의 표시값을 역산하지 않는다.
- 주석에 없는 관계는 UNKNOWN일 수 있다.
- “주석에 없음”을 “그림에 없음”으로 바꾸지 않는다.

CLEVR:

- image filename/index와 scene을 정확히 연결한다.
- 좌우 순번은 실제 이미지 x 좌표를 사용한다.
- x tie 또는 불명확한 identity는 UNKNOWN으로 둔다.
- 객체의 부분 가림 등 주석과 가시성의 차이를 범위에 기록한다.

차트:

- 원 CSV·행/열·series/category·숫자·단위·그림을 연결한다.
- OCR에 같은 숫자가 있다는 것만으로 해당 막대의 값이라고 인증하지 않는다.
- 프로그램이 새 정답 설명을 작성해 SFT 목표로 넣지 않는다.

### 7.4 claim ledger

프로그램은 생성이 끝난 뒤 OBS 주장에는 o0, o1, T1 주장에는 t0, t1 등의 짧은 ID를 부여한다.
ordinal은 각 실제 출력의 원문 순서다. NOTE 입력 안에서 stage까지 포함한 ID가 유일해야 한다.
긴 SHA 문자열을 모델이 다시 복사하도록 요구하지 않는다.
ID의 유일성은 ledger의 generation hash와 함께 검증한다.

    schema: v17-claim-ledger-v1
    problem_id
    stage: OBS | T1
    generation_sha256
    original_text_sha256
    original_token_ids_sha256
    image_sha256
    verifier_sha256
    claims:
      - claim_id
        source_json_path
        original_entity
        original_value
        original_byte_spans
        canonical_fact_key
        verdict: VERIFIED | CONTRADICTED | UNKNOWN
        reason
        verifier_evidence_sha256

Unicode 문자 위치와 UTF-8 byte 위치를 구별한다.
JSON duplicate key·과도한 depth·비유한 숫자·제어 token을 거부한다.
동일 entity의 충돌 주장을 정상 단일 claim으로 합치지 않는다.
실제 원문을 고쳐서 파싱에 성공시키지 않는다.

### 7.5 OBS 보상

v17 기본안은 **짧은 구조화 관측 전체에 대한 보상**이다.
N에서는 검증된 교정 span을 학습하지만, OBS PG를 오류 위치의 인과적 credit이라고 설명하지 않는다.
새 token별 process RL 알고리즘을 동시에 도입하지 않는다.

비공개 adapter가 제공하는 검증 가능한 target facts를 F라 한다.
모델이 실제 생성한 정규화된 claim을 G라 한다.

- TP: F와 일치하는 VERIFIED claim.
- FP: 같은 scope에서 명확히 CONTRADICTED인 claim.
- FN: F 중 G에서 맞게 표현되지 않은 사실.
- UNKNOWN: 오류로 인증하지 않고 별도로 센다.
- 중복 claim은 TP를 늘리지 않는다.
- scope 밖 주장으로 보상을 늘릴 수 없다.

    r_obs = 2*TP / (2*TP + FP + FN)

분모가 0이면 OBS_NO_VERIFIABLE_TARGET이며 PG 제외다.
전부 UNKNOWN이고 F가 비어 있지 않으면 보상 0이며, UNKNOWN을 거짓으로 판정했다는 뜻은 아니다.
malformed/length 종료의 보상·PG 여부는 별도 reason으로 고정한다.
length 종료는 기존 원문 불완전성 정책에 따라 PG 제외한다.
정상 종료했으나 schema가 틀린 경우 보상 0으로 기록하되 교정 성공으로 세지 않는다.

이 F1은 adapter가 인증하는 부분 범위의 지표다.
그림 전체 이해도나 T1 추론의 정확도를 인증하지 않는다.
geometry의 불완전 주석과 graph의 closed-world를 같은 의미로 해석하지 않는다.

OBS 음의 advantage에는 기존 확률 0.5 마스크를 적용하지 않는다.
오류 관측 신호가 거의 모두 제거된 문제를 이 경로에서 먼저 고친다.
응답 수준 보상이므로 함께 맞았던 claim에도 음의 credit이 갈 수 있다는 한계를 기록한다.

## 8. NOTE — 실제 오류와 수정의 연결

### 8.1 대상 결정

- T1 오답: 노트 후보.
- T1 정답이어도 OBS 또는 검증 가능한 T1 claim에 오류가 있으면 노트 후보.
- T1 잘림: 신뢰할 수 있는 완결 claim 또는 OBS 오류가 있으면 제한된 교정 후보.
- T1 잘림 전체를 틀린 풀이로 인증하지 않는다.
- claim 근거가 없는 경우 자유 교훈을 만들 수는 있으나 N 교정 학습으로 세지 않는다.
- 모든 후보·skip 사유를 분모에 남긴다.

NOTE 적격은 reward_v16b의 pg 플래그와 독립적으로 계산한다.
기존 build_note_prompt_v16b의 `if not pg: SKIP` 분기를 v17에 가져오지 않는다.
잘린 T1도 실제로 완결된 검증 오류 또는 독립 OBS 오류가 있으면 NOTE와 안전한 재풀이에 진입할 수 있다.
그렇다고 잘린 T1 자체를 PG나 A1의 정상 정답 풀이로 수용하지는 않는다.
완전한 입력이 문맥 상한을 넘으면 NOTE_CONTEXT_BLOCKED로 남긴다.

NOTE 입력은 전체 private ledger를 직렬화하지 않고 다음 허용 필드만 투영한다.

    claim_id, stage, original_entity, original_value, original_quote, verdict

검증 실패의 세부 reason, canonical_fact_key, 원 주석 경로, 정답 fact/value,
private evidence와 hash는 모델 입력에 넣지 않는다.
원문 질문·이미지·자기 풀이와 채점 뒤 허용된 gold는 별도 필드로 구성한다.
OBS/T1 양쪽 ordinal 0이 있어도 o0와 t0로 정확히 구분한다.

### 8.2 모델 출력

    {
      "repairs": [
        {
          "claim_id": "o0",
          "corrected_value": "7"
        }
      ],
      "lesson": "Check the printed label before applying the relation."
    }

최초 v17은 노트당 repair 최대 한 개로 제한한다.
복잡한 cause label·rounding·문자 offset·old_value 재출력을 제거한다.
이는 노트가 참이라는 검증을 완화하는 것이 아니라 출력 계약을 줄이는 것이다.
old_value는 실제 자기 원문으로 입력에만 존재한다.

### 8.3 VERIFIED_REPAIR 조건

다음 조건을 모두 만족해야 한다.

1. claim ID가 해당 부모·해당 generation의 봉인된 ledger에 존재한다.
2. 원래 주장이 실제 생성 원문에 있다.
3. 인용·가정·철회된 문장이 아니라 현재 단언으로 연결된다.
4. 원래 값의 판정은 CONTRADICTED다.
5. 수정은 같은 entity·속성·단위·scope에 관한 것이다.
6. 모델이 생성한 corrected_value가 독립 verifier에서 VERIFIED다.
7. NOTE 생성 원문·token·정책판·입력 hash가 일치한다.

다음은 교정 성공이 아니다.

- 원래 맞던 내용을 다시 적음.
- 이전에 말하지 않은 사실을 새로 나열함.
- UNKNOWN을 새 사실로 보충함.
- 마지막 답만 gold에 맞춤.
- 다른 entity로 바꿔 쉬운 사실을 말함.
- 프로그램이 생성한 수정값을 모델 출력처럼 저장함.

UNKNOWN→VERIFIED와 새 사실 습득은 별도 관측 지표로 기록할 수 있다.
VERIFIED_REPAIR 수에 합치지 않는다.

### 8.4 산술 교정

기존 산술 helper는 실제 단언 등식에만 사용한다.
숫자 하나나 식 조각만 인용한 것을 잘못된 계산으로 만들지 않는다.
수정은 같은 피연산자·연산·단위의 결과를 다뤄야 한다.
문제의 전문 규칙 적용까지 산술 검사로 인증하지 않는다.
산술은 보조 경로이며 대부분의 오답에 등식이 존재한다고 가정하지 않는다.

### 8.5 N·교훈·C를 독립적으로 판정

    n_accepted
    verified_repair_count
    lesson_accepted
    lesson_leak_reasons
    resolve_allowed
    c_accepted

위 필드를 하나의 accepted로 합치지 않는다.

- N 성공 / 교훈 누설: N의 확인된 수정은 학습, 재풀이 차단.
- N 성공 / C 오답: N 학습, C 학습 제외.
- N 실패 / 안전한 교훈 / C 정답: C는 자체 검사를 통과한 경우 수용 가능.
- N 실패 / C 실패: 성공 사례를 만들어 넣지 않음.
- error:null 사실 나열: N 교정 성공 아님.

현재 v16b에서도 N과 C는 코드상 완전히 동일한 gate가 아니다.
v17 변경을 “처음으로 N을 C와 분리했다”고 과장하지 않는다.
주요 수정은 실제 교정 계약·입력·마스크·독립 receipt를 제대로 연결하는 것이다.

### 8.6 N 학습 마스크

학습 token은 실제 NOTE 출력의 검증된 corrected_value에 연결한다.
v17 기본 N 손실은 content-only다. claim ID·key·구분자·JSON 괄호에는 SFT 손실을 주지 않는다.
corrected_value의 값 영역만 mask에 포함하고, 형식 token 학습을 별도 옵션으로 추가하지 않는다.
따옴표와 값을 함께 담은 경계 token은 완전 포함 조건에 따라 제외될 수 있다.
verified content token이 0이면 N 수용 불가다.

자유 lesson, 미확정 설명, 잘못된 이전 값, EOS/control token은 제외한다.
byte span에 완전히 포함되는 실제 token만 학습한다.
경계를 가로지르는 token을 임의로 쪼개거나 새 token으로 바꾸지 않는다.

## 9. A1·C·준비 SFT — 학습량을 늘리되 인증 범위를 명시

### 9.1 공통 검사기

v17_rows.verify_solution_v17을 추가한다.
모든 검사를 수행한 뒤 수용 여부와 실패 목록을 만든다.
첫 실패 하나에서 반환해 뒤의 모순 검사를 누락하지 않는다.

필수 검사:

1. 실제 생성 원문·token ID·정책판·입력·이미지 hash 연결.
2. 정상 종료.
3. 고정 문맥·출력 상한.
4. 반복 검사.
5. 기존 학습 채점기의 정답 판정.
6. 단일 최종답과 추출값 일관성.
7. 답 부정·복수 결론.
8. 검사 가능한 관측·산술·과정 모순.
9. 최종답 외의 설명 내용 존재.
10. C에만 교훈 누설·복사·참고문구·이미지 회피 검사.

256토큰 최소 조건을 제거한다.
4,096토큰이라는 과거 SFT 전용 상한도 v17에서 8,192 계약으로 통일한다.
길이 상한까지 썼다는 이유로 반복·모순 검사를 면제하지 않는다.

### 9.2 답만 있는 목표 제외

final-answer span을 제외한 텍스트를 검사한다.
빈 글·선택지 문자 반복·정답 문장 재진술만 있으면 ANSWER_ONLY로 제외한다.
지원 영역에서는 적어도 하나의 확인 가능한 관측·관계·연산을 요구한다.
무주석 영역에서는 non-final 설명의 존재와 검출 가능한 모순 부재까지만 검사할 수 있다.

일반 자연어 설명의 의미를 정규식으로 완전 검증했다고 주장하지 않는다.
길이가 길거나 because가 있다는 이유만으로 정확한 reasoning으로 인증하지 않는다.

### 9.3 두 가지 학습 범위

| 행 | 학습 범위 | 정확성 표기 |
|---|---|---|
| A1 | 수용한 자기 첫 풀이의 non-EOS 원문 | RESULT_CONDITIONED_PARTIALLY_CHECKED |
| C | 수용한 자기 재풀이의 non-EOS 원문 | RESULT_CONDITIONED_PARTIALLY_CHECKED |
| N | 독립적으로 확인한 교정 span | VERIFIED_REPAIR_CONTENT |
| PREP_SELF | 수용한 F0 자기 풀이 | RESULT_CONDITIONED_PARTIALLY_CHECKED |
| PREP_HUMAN | 출처가 확인된 데이터셋 풀이 | DATASET_AUTHORED, 자체 전수 사람 인증 아님 |

A1/C는 최종답이 맞고 검사 가능한 모순이 없는 자기 풀이를 학습한다.
풀이 전체를 정확하다고 증명한 것은 아니다.
이 위험은 기존 모델의 성공 행동을 학습하는 목적과 함께 명시한다.
새 N의 엄격한 교정 인증과 같은 지표로 합치지 않는다.

### 9.4 추가 생성 없는 A1

- T1 완료 때 동일 출력으로 RL 행과 A1 후보를 만든다.
- A1 때문에 T1을 다시 생성하지 않는다.
- A1 입력은 원래 P0다.
- 정답·노트·OBS 결과를 뒤늦게 A1 입력에 붙이지 않는다.
- row_id는 부모 ID와 kind로 유일하게 만든다.
- 같은 부모의 RL+A1은 의도된 이중 손실이며 중복 부모 생성이 아니다.
- 이미 소비한 A1을 재개 후 다시 학습하지 않는다.

### 9.5 C의 입력 전환

RESOLVE 생성 입력에는 안전한 lesson이 있다.
C 학습 입력은 lesson 없는 P0다.
따라서 생성 입력 hash와 SFT 입력 hash를 별도 보존한다.
이 전환을 같은 확률의 on-policy RL로 취급하지 않는다.
C는 raw CE의 자기생성 SFT다.

### 9.6 교훈 누설 검사

정답 문자·정답 숫자·보기 식별 정보·수정값의 직접 전달을 검사한다.
문제와 모든 보기에 이미 있는 일반 단어를 정답 누설로 오인하는 사례는 별도 회귀로 다룬다.
공통 단어 면제만으로 숫자·정답 선택 누설을 우회할 수 없어야 한다.
기존 누설 규칙을 통째로 끄거나 모든 stem 차단을 제거하지 않는다.
명확한 안전/누설/미확정으로 구분하고 미확정이면 재풀이를 허용하지 않는다.

## 10. RL·SFT·원본 보존 목적함수

### 10.1 기호와 고정 분모

- B = 16: 본학습 부모 분모.
- T = 1,024: PG token scale.
- n_i: 해당 응답의 실제 action token 수.
- A_i = r_i − b_i: 생성 전에 정한 baseline을 뺀 advantage.
- q: 실제 sampling 처리를 적용한 전체 어휘 분포.
- p: raw next-token 분포.
- μ: 해당 응답 생성 당시 저장한 log q.
- sg: stop-gradient.

마지막 8문항과 마무리 슬롯도 B=16을 유지한다.
없는 A1/C/N 행을 제외한 개수로 재정규화하지 않는다.
PREP는 별도 단계로 기존 actual_rows 분모를 유지한다. main의 고정 부모 분모와 섞지 않는다.

### 10.2 제안 손실

    L = L_T1 + L_OBS
        + 0.20 L_A1
        + 0.35 L_C
        + 0.05 L_N
        + 0.02 L_F0_positions

위 계수는 **초기 설계값**이다.
과거 실측 최적값 또는 65점 달성값이 아니다.
첫 실제 기능 시험에서 유한성·기여·비용을 확인한 후 본학습 protocol을 봉인한다.
본학습 결과를 보고 계수를 자동 조정하지 않는다.

### 10.3 PG

    L_K = - alpha_K / (B*T)
          * sum_i c_i * A_i
          * sum_t sg(min(2, exp(logq_current_it - logq_behavior_it)))
                  * m_it * logq_current_it

    alpha_T1 = 1
    alpha_OBS = 0.25
    c_i = min(1, 2048/n_i)

T1 양의 advantage: 실제로 표집한 전체 action token.
T1 음의 advantage: 기존 확률 ≤0.5 마스크 유지.
OBS 양·음의 advantage: 해당 짧은 관측 응답의 action token.
OBS에는 확률 마스크를 적용하지 않는다.
PG에는 실제로 표집한 유효 terminal EOS도 포함한다. 이는 기존 PG 범위를 유지하는 결정이다.
n_i도 terminal을 포함한 원래 completion_token_ids의 길이다.
그 밖의 금지 control/image/video token은 거절하고, 존재하지 않은 EOS를 PG에 추가하지 않는다.
SFT의 EOS 제외와 PG의 terminal 포함을 서로 다른 mask로 검사한다.

T1 마스크, 길이 cap, importance clipping, 지연 정책 때문에 이 손실은 명시적 surrogate다.
unbiased REINFORCE 또는 표준 GRPO라고 부르지 않는다.
token별 비율은 전체 trajectory prefix의 분포 차이를 완전히 보정하지 않는다.
따라서 명칭은 편향된 off-policy surrogate이며, 마스크만이 유일한 편향 원인인 것도 아니다.
일반 T1 전체의 음의 마스크 제거는 이번 기본안에 포함하지 않는다.
검증된 관측 오류 경로를 먼저 바꾸고, 일반 추론 전체를 강하게 억제하는 변경을 동시에 넣지 않기 위해서다.
그 결과 정상 종료 오답에 가는 직접 신호는 약하다. v16b 실측으로 T1 음의 후보 684,059토큰 중
마스크 뒤 107,294토큰(15.7%)만 남았다. 이 비율을 묶음마다 기록하고(§12.1) 위험으로 명시한다(§22).

### 10.3.1 반복 확인 풀이(LOOP_CONFIRMED)의 음의 PG

`[사용자 2026-11-16]` 검토 보강 지시로 v17 기본안에 넣는다. 보상 규칙의 변경이며 RL·자기생성 학습이라는 학습 방식은 그대로다.

근거(모두 저장 기록의 실측):

| 관측 | 수치 |
|---|---|
| v16b 첫 풀이 중 PG 제외(길이 상한 251 + UNVERIFIABLE 7) | 258개(8.6%)가 T1 생성 토큰의 61.34% 사용, 직접 학습 신호 0 |
| 그중 기존 반복 검사(`contract.is_loop`)에 걸린 것 | 길이 상한 251개 중 165개(66%) |
| 정상 종료 T1에서 같은 검사의 오검출 | 2,749개 중 0개 |
| v16b 학습 중 8,192 도달 비율 | 처음 24묶음 7.3% → 이후 구간 약 11% |
| 원본 B2 평가 출력 | 4,116,922토큰 중 72%가 32,768 도달 90문항 |
| v15 같은 450문항 분해 | 원본이 잘렸던 46문항 1→9, 정상 종료 404문항 274→271 |
| v13 반복 구간 감점 | 32K 도달 90→27, 점수 555(원본 558과 구별 불가) |

규칙:

    reward_v17(T1):
      finish_reason=length AND is_loop(text)  → reward 0, reason LOOP_CONFIRMED, pg True
      finish_reason=stop   AND is_loop(text)  → reward 0, reason LOOP_CONFIRMED, pg True
      finish_reason=length AND NOT is_loop    → reward 0, reason LENGTH, pg False   # 기존과 같음
      그 밖                                    → 기존 v16b 보상 규칙

- 반복 검사는 기존 `contract.is_loop`(마지막 3,000자, 압축률 0.12 미만)를 그대로 쓰고 문턱을 바꾸지 않는다.
- advantage는 기존과 같이 `A = 0 − b`(풀기 전에 봉인한 기준값)다. 따라서 음수다.
- PG 토큰 범위 m: 감지에 쓴 마지막 3,000자에 UTF-8 byte가 완전히 포함되는 실제 생성 토큰만. 경계를 가로지르는 토큰은 제외한다.
- 이 범위에는 T1의 확률 ≤0.5 마스크를 적용하지 않는다. 반복 토큰은 대부분 고확률이라, 마스크를 쓰면 억제해야 할 토큰이 거의 지워진다(관측 RL에서 마스크를 뺀 것과 같은 이유).
- 행 가중치 `c_i = min(1, 2048/n_loop)`, n_loop는 위 범위의 토큰 수다. 분모는 T1 채널의 B·T를 그대로 쓴다.
- importance 비율·clip·음의 행 |log-ratio|>0.3 제외·HOLD 규칙은 일반 T1 음의 행과 같다.
- LOOP_CONFIRMED 풀이는 A1·C·PREP_SELF 목표가 될 수 없다. NOTE 적격은 §8.1 그대로다(검증된 완결 주장이 있을 때만).
- 관측·노트·재풀이 생성에는 이 규칙을 적용하지 않는다(C는 기존 반복 검사로 이미 거절).
- 묶음마다 LOOP_CONFIRMED 수, 감점 토큰 수, LENGTH(비반복) 수, 8,192 도달 비율을 기록한다(§12.1).

기대와 한계:

- 목적은 반복으로 낭비되는 생성 계산과 평가 시간·KV 위험을 줄이는 것이다. v13 근거로 보면 이것만으로 점수가 오른다는 근거는 없다.
- 반복 억제가 지나치게 짧은 풀이로 이어질 수 있으므로 §12.4의 단답 증가 규칙과 함께 본다.
- 반복이 확인되지 않은 잘림(v16b 86개)은 계속 신호가 없다. 이를 오답으로 인증하지 않는다.

### 10.4 확률 정의

    processed_logits = (raw_logits - 1.5 * seen_generated_token) / 0.7
    q = softmax(processed_logits)

seen_generated_token은 해당 생성의 이전 token만 포함한다.
prompt의 단어를 presence penalty의 생성 이력으로 섞지 않는다.
chunk 경계에서도 이전 생성 token 상태를 유지한다.

양수·음수 RL은 같은 q를 사용한다.
음수에만 raw p를 쓰거나 옛 top-k support 확률을 섞지 않는다.
top-k/top-p 절단이 없으므로 v15의 singleton support 문제를 되돌리지 않는다.

### 10.5 baseline과 지연 보정

기존 v16b의 생성 입장 시점 baseline receipt를 재사용한다.
현재 응답 결과를 본 뒤 baseline을 바꾸지 않는다.
지원되지 않는 OBS prior를 prep에서 측정했다고 만들지 않는다.
OBS cold-start는 실제 이전 적격 관측 기록이 생길 때까지 명시적으로 기록한다.

기존 baseline의 수식과 시작 동작은 다음처럼 고정한다.

    source_mean = (8 * prep_source_prior + sum(last_64_committed_source_rewards))
                  / (8 + source_history_count)
    b_T1 = source_mean                                  # 개별 F0 prior 없음
    b_T1 = (2 * source_mean + exact_item_f0_correct) / 3 # exact prior 있음

해당 출처의 prep prior가 없으면 prep overall을 사용한다.
개별 prior는 허용된 학습 자료의 동일 ID·질문·보기·이미지에 연결된 과거 F0 기록만 사용한다.
OBS는 층별 최근 32개와 pseudo_count 8을 사용한다.
그 층의 prep 관측이 없으면 첫 committed OBS 묶음 평균을 initial로 고정한다.
그 이전에는 baseline=null, zero_advantage=true로 두어 첫 입장 묶음의 OBS PG를 0으로 한다.
OBS 보상은 새 F1 계약으로 계산하므로 과거 v16b OBS 점수를 초기 이력으로 섞지 않는다.
cutoff는 기존 η=1 계약의 block_index−2이며 현재 묶음의 결과를 소급 사용하지 않는다.

기존 IS drift 문턱을 버전별 계약에 유지한다.

- 음의 행 평균 절대 log-ratio >0.3: 해당 PG 제외.
- 묶음 평균 >0.5: 공동 갱신 HOLD.
- ratio clipping 비율 >0.10: HOLD.
- mixed-version 응답이 한 개라도 있으면 HOLD. 이 항목은 v16b의 5% 허용을 v17에서 폐기한다.
- lag-zero replay 평균 오차 >0.05: HOLD.

HOLD는 이미 계산한 A1/C/N/KL 기울기도 모두 지운다.
이를 “RL 대신 SFT로 성공”이라고 보고하지 않는다.
HOLD가 반복되면 원인 수정 없이 재실행하지 않는다.
mixed 검사는 PG 행뿐 아니라 수용할 A1/C/N의 생성과 인용한 자기 원문까지 확인한다.
혼합된 응답을 SFT로 우회 수용하거나 가짜 단일 policy hash를 붙이지 않는다.
가능한 한 backward 전에 검사하며, 뒤늦게 발견되면 공동 gradient를 전부 폐기한다.

### 10.6 SFT

    L_K = (1/B) * sum_i [
            - sum_t mask_it * log p_current(action_it | training_input_i, prefix_it)
              / sum_t mask_it
          ]

본학습 K는 A1, C, N이다.
마스크 token 수가 0이면 행 수용 불가다.
raw CE로 바꾸는 것은 v16b의 처리 후 자기 SFT와 다른 목적함수다.
새 protocol에 probability_space_sft=raw-full-vocab-v1를 명시한다.
같은 raw CE 원칙을 PREP_SELF와 PREP_HUMAN에도 적용한다.
PREP_SELF·C·N은 기존 processed-q CE에서 변경되고, PREP_HUMAN은 기존 raw CE를 유지한다.

기존 C/N은 각각 행당 0.35/0.05를 합했다. 같은 수용 행이라면 새 고정 분모 16은 각 기여를 16배 줄인다.
이는 수용량 증가에 대비한 새 가중치 설계이며, 기존과 같은 세기를 유지하는 단순 수정이 아니다.
채널별 gradient와 실제 수용량을 확인하지 않고 RL 중심 비중이 달성됐다고 인증하지 않는다.

준비 단계는 다음 별도 손실을 사용한다.

    b_prep = 실제 현재 준비 batch의 행 수 (1..16)
    L_prep = sum(mean_raw_CE(row) for row in prep_batch) / b_prep
             + 0.5 * sum(mean_first_4_positions_F0_KL(row)) / b_prep

PREP_SELF/HUMAN 계수는 각각 1이며, 마지막 준비 batch도 실제 행 수로 나눈다.
준비 단계의 첫 네 위치 anchor는 유지하고 본학습 64위치 항을 중복 부과하지 않는다.
준비 SELF 상한 160행·HUMAN 상한 128행은 유지한다. 변경된 수용 기준으로 새 행 목록을 봉인한다.
사람 풀이의 기존 32토큰 하한과 출처 검사는 유지하며, 자기 풀이의 256토큰 하한은 §9로 대체한다.

T1에서 RL과 A1을 같이 사용한다는 사실을 보고한다.
SFT 수용 수가 늘면 실제 총 기여도는 늘지만, 행 수에 따른 무제한 상대 가중치 증폭을 고정 분모로 제한한다.
loss scalar와 token 수만으로 RL이 기울기의 대부분이라고 인증하지 않는다.

### 10.7 원본 보존: 응답 전체 64위치

정확한 모든 token·전체 어휘 KL은 긴 T1에서 계산량이 크다.
v17 기본안은 **응답 전반에 분산된 최대 64위치의 전체 어휘 KL**이다.
이를 모든 token의 exact KL이라고 부르지 않는다.

각 T1의 n개 action 위치를 최대 64개의 연속 구간으로 나눈다.
구간마다 고정 seed·부모 ID·protocol salt로 한 위치를 선택한다.
선택은 정오·현재 token 확률·노트 내용과 독립이다.
각 구간의 길이/n으로 가중한다.
n≤64이면 모든 위치를 사용한다.

    L_F0_positions = (1/B) * sum_i sum_bin
        (bin_size / n_i)
        * KL(p_F0(. | actual_prefix_at_selected_position)
             || p_current(. | same_prefix))

선택 위치와 가중치를 receipt에 저장한다.
구간 내 hash 표본은 고정 실행에서 근사 보존항이며, 전체 응답의 완전 보존을 인증하지 않는다.
첫 네 token 전용 anchor는 이 항으로 대체하고 중복 부과하지 않는다.
이는 본학습에 대한 대체이며 준비 단계는 §10.6의 별도 식을 따른다.
본학습 계수 0.02는 기존 0.5보다 25배 작다. 위치 범위가 넓어진 것과 보존 강도가 커진 것은 다르다.
0.02가 최적이라는 근거는 없으며 실제 기여와 비용을 확인할 초기 제안값이다.
원문·정책 무결성이 유효하고 action 길이가 양수인 모든 T1에 부모당 한 번 적용한다.
정답·오답·잘림·advantage=0·PG 제외를 모두 포함하고 A1이 추가돼도 두 번 부과하지 않는다.
빈 응답은 KL_EMPTY_RESPONSE로 기록한다. 원문/hash 손상은 무결성 오류로 종료한다.
mixed 응답이 있으면 §10.5에 따라 공동 HOLD하며 정상 제외로 처리하지 않는다.
KL만 존재하고 task 신호가 없으면 optimizer step을 하지 않는다.

### 10.8 F0 계산과 비용

같은 기저 모델에서 adapter를 끄고 원본 Merger를 선택해 F0 hidden을 계산한다.
두 번째 4B 모델을 별도로 적재하지 않는다.
mergers_original과 원본 hash 검사를 재사용한다.
예외 발생 때 adapter·route·train/eval·requires_grad를 원상 복구한다.

선택 위치만 vocabulary projection을 계산하므로 head 계산량을 줄인다.
그러나 각 위치의 prefix를 처리하는 F0 forward 비용은 남는다.
64위치라는 이유로 64token만 forward하면 되는 것이 아니다.
F0 hidden은 detach하고, 원본 파라미터에 gradient가 생기지 않음을 검사한다.
현재 hidden은 PG와 가능한 범위에서 공유한다.
학습이 진행되면 달라지는 prefix에 대해 다른 응답의 F0 cache를 재사용하지 않는다.

최대 길이 FULL 행으로 메모리·시간을 측정한다.
측정 실패 때 자동으로 KL을 끄거나 첫 네 token으로 되돌리지 않는다.
필요하면 실행 전에 새 protocol을 봉인한다.

### 10.9 optimizer와 slot

학습률 초기 설계는 기존값을 유지한다.

    L: 5e-5
    M: 1e-5
    V: 1e-5
    warmup_slots: 4
    grad_clip_per_group: 1.0

P를 실제 준비 batch 수라 하면 S=P+188+3이다.
slot은 optimizer step 여부와 무관하게 한 번씩 소비한다.
updates는 실제 optimizer.step 수다.

기존 선형 일정은 마지막 slot에서 양의 LR, 모든 slot 소비 뒤 0이다.
마지막 세 slot이 비었으면 마지막 실제 parameter update LR은 0이 아닐 수 있다.
이를 오류나 완료 증거로 혼동하지 않는다.
lr_before, lr_after, optimizer_step을 매 slot 기록한다.

## 11. 행·encoder·영수증 인터페이스

### 11.1 공통 행

    schema: error-note-v17-row-v1
    row_id: problem_id + ":" + kind
    kind: RL | OBS | A1 | C | N | PREP_SELF | PREP_HUMAN
    problem_id / source / layer / block_index
    prompt / target
    route: FULL | LM
    image_paths / image_sha256
    generation
    generation_prompt_sha256
    training_prompt_sha256
    admission_receipt_sha256
    verifier_receipt_sha256
    loss_probability_space
    loss_token_mask
    verified_content_token_mask
    knowledge_status

RL/OBS에만 reward, baseline, advantage, behavior logq, baseline receipt가 필요하다.
HUMAN은 모델 생성 token이라고 위장하지 않는다.
각 kind의 필수·금지 필드를 validator가 검사한다.

### 11.2 함수 계약

| 함수 | 입력 | 출력 / 책임 |
|---|---|---|
| build_obs_request_v17 | public item, generic source schema | gold·private facts 없는 요청 |
| audit_observation_v17 | raw generation, private verifier | claim ledger, reward receipt |
| build_note_request_v17 | 자기 원문·ledger·채점 뒤 gold | private annotation 없는 NOTE |
| verify_note_v17 | raw NOTE, 원 ledger, private verifier | N/lesson/resolve 독립 판정 |
| verify_solution_v17 | T1 또는 RESOLVE, 원문 binding, grading, 검증 근거 | 모든 검사 결과와 수용 판정 |
| a1_row_v17 | T1, 수용 receipt | P0→T1 원문 SFT 또는 없음 |
| c_row_v17 | RESOLVE, 안전 lesson receipt, 수용 receipt | P0→RESOLVE 원문 SFT 또는 없음 |
| n_row_v17 | NOTE, repair receipt | 검증 span mask SFT 또는 없음 |
| encode_v17_row | row, processor, model config | token/byte/P0/route 검증된 tensor 입력 |
| backward_v17_block | encoded rows, 봉인된 update contract | channel stats, task signal, HOLD |
| reserve_request_v17 | 준비된 요청, 실제 학습 KV, live reservations | 수용/대기/용량 오류, 예약 영수증 |
| calibrate_eval_v17 | 비벤치마크 probe, 실제8칸 엔진 통계 | 처리량 범위·coverage·watchdog 계산 영수증 |
| resume_eval_v17 | 기존 등록·START·run, 봉인된 재개 인자 | 등록 선검사 후 미완료 ID만 --resume |
| recover_final_v17 | 원자 FINAL_RAW, 입력·모델 binding | 새 생성 없이 동일 V8 sample commit |

함수 이름은 이 계획에서 구현할 API다.
현재 코드에 이미 존재하는 함수라고 보고하지 않는다.

### 11.3 encoder 주의점

현재 encode_v16b_row는 N만 span mask를 다시 계산한다.
C/PREP_SELF는 모든 non-EOS token을 True로 재구성한다.
v17은 각 kind의 mask mode를 명시하고 같은 규칙으로 독립 재계산한다.
행 작성기만 바꾸고 기존 encoder에 넣으면 안 된다.

- A1/C/PREP_SELF: 전체 non-EOS mask.
- N: 검증된 실제 byte span의 완전 포함 token.
- PREP_HUMAN: 출처가 인증된 원문 목표, control/EOS 정책 고정.
- RL/OBS: sampling action mask와 PG 필터를 SFT mask와 분리.

모델 생성물은 저장된 sampled token IDs를 유지하고, 허용 terminal을 처리한 decode 결과와 raw text를 대조한다.
decode 후 재encode가 항상 같은 token 배열이라는 가정을 하지 않는다.
token/byte span의 일치는 원 token에 대해 검사한다.
모델 생성물에 EOS·정답 문자열을 붙여서 통과시키지 않는다.
PREP_HUMAN만 기존 encoder대로 원 텍스트를 tokenize하고 masked boundary EOS를 구성할 수 있다.
그 EOS에는 손실이 없으며 모델이 생성한 token이라고 표시하지 않는다.

### 11.4 callback 연결 순서

다음은 새 알고리즘의 의사코드다. 현재 함수가 이미 이대로 구현됐다는 의미는 아니다.

    complete(T1):
        원문·정책·입력 binding 검사
        기존 grade_single로 답 판정
        v17 reward receipt와 T1 claim ledger 봉인
        PG 여부와 무관하게 NOTE 후보 사유 기록
        verify_solution_v17로 A1의 모든 검사 결과 기록

    complete(OBS):
        public scope와 private adapter binding 검사
        실제 주장 ledger와 부분 범위 F1 receipt 봉인
        T1 정오와 별도로 관측 오류 수 기록

    request(NOTE):
        해당 부모의 T1/OBS 완료 receipt를 사용
        NOTE eligibility가 없으면 명시적 SKIPPED 기록
        있으면 whitelist 자기 주장 + 허용 gold로 1회 요청

    complete(NOTE):
        repair, lesson, resolve_allowed를 독립 판정
        N content-only byte span과 token mask를 원문에 연결

    request(RESOLVE):
        lesson 안전 판정이 확정된 부모만 P0+lesson으로 최대 1회 요청

    complete(RESOLVE):
        verify_solution_v17의 모든 검사 수행
        P0 학습 입력과 실제 생성 입력의 hash를 각각 보존

    build_rows(slot):
        예정 T1마다 RL 행 유지 (pg=false도 이유를 보존)
        같은 T1의 A1 수용 행 추가
        예정 OBS, 지연 N, 지연 C만 해당 slot 계약대로 추가
        mixed/source/중복 소비 검사 후 encoder에 전달

    update(slot):
        공유 가능한 RL+A1 계산 묶음과 별도 행을 구성
        채널별 고정 분모로 backward
        HOLD면 모든 gradient 폐기, 아니면 유효 task 신호로 step
        slot·scheduler·소비 원장·guard·gate를 일관되게 저장

### 11.5 새 protocol 필수 식별자

    version: error-note-v17
    row_schema: error-note-v17-row-v1
    claim_schema: v17-claim-ledger-v1
    capability_schema: v17-capability-v1
    probability_space_rl: processed-full-vocab-v1
    probability_space_sft: raw-full-vocab-v1
    note_mask: verified-corrected-value-content-only-v1
    obs_reward: bounded-scope-f1-v1
    negative_t1_mask: probability-le-0.5
    negative_obs_mask: all-sampled-actions
    loop_signal: confirmed-loop-negative-pg-v1
    loop_detector: contract.is_loop(window=3000, ratio=0.12)
    loop_token_scope: detected-window-no-probability-mask
    eval_adapter_serving: unmerged-lora
    comparison_strata: f0-termination-state-v1
    mixed_policy_action: hold-any-mixed-source
    main_parent_denominator: 16
    prep_denominator: actual-rows
    f0_main_positions: stratified-64-bin-weighted-v1
    f0_prep_positions: first-4
    signal_gate_supported_parents: 32
    signal_gate_verified_repair_parents: 3
    training_concurrency_ceiling: 16
    training_kv_admission: full-request-reservation-v1
    short_drift_warning_delta: 0.05
    short_drift_stop_min_delta: 0.15
    short_drift_stop_se_multiplier: 4
    eval_concurrency: 8
    eval_kv_capacity_margin: 0.10
    eval_refill: continuous
    eval_resume: immutable-completed-same-registration-v1
    eval_watchdog_source: sealed-throughput-calibration
    total_timeout_seconds: null
    training_elapsed_cutoff: null
    target_total_hours: [14, 17]

값은 builder 한 곳에서 정의하고 CLI·row validator·trainer·export·resume가 이를 검사한다.
식별 문자열과 수치가 문서와 어긋나면 문서·코드 중 하나를 몰래 우선하지 말고 동결 전에 해결한다.
명세서 hash와 source/data/order/protocol hash를 실행 manifest에 함께 남긴다.

## 12. 신호 관문 — 교정 0을 끝까지 방치하지 않기

### 12.1 보고할 단계별 분모

매 묶음 다음 수를 기록한다.

    parents_planned / parents_consumed
    T1_generated / graded / PG_positive / PG_negative / PG_excluded
    OBS_eligible / requested / completed / schema_valid
    claims_total / verified / contradicted / unknown
    NOTE_requested / completed / verified_repairs
    N_rows / N_verified_content_tokens
    lesson_safe / lesson_leak / lesson_unknown
    RESOLVE_requested / completed / correct
    A1_candidates / A1_accepted
    C_candidates / C_accepted
    rejection_reasons_all
    length_n / short_lt32 / answer_only / no_final / truncated
    length_histogram / length_p10 / median / p90 / p95
    first24_short_count_and_n / recent12_short_count_and_n / short_rate_delta
    T1_negative_candidate_tokens / T1_negative_retained_tokens / retained_ratio
    LOOP_CONFIRMED / loop_tokens_penalized / LENGTH_nonloop_excluded / cap_rate_8192

길이 histogram은 `<32 / 32..127 / 128..255 / 256..1023 / 1024..4095 / 4096..8191 / >=8192`다.
finish_reason=length는 마지막 histogram 구간과 별도로 센다.
출처별 분모도 기록하며 정확도·길이 통계의 서로 다른 표본 수를 합치지 않는다.

수용된 N만 분모로 써서 성공률을 높여 보이지 않는다.
잘림·무신호 문항의 생성 token·시간도 별도로 합산한다.
전체 계산량과 직접 학습 신호의 차이를 보고한다.

### 12.2 본학습 확대 전 gate

새 모델 생성 전에 판정 기준을 봉인한다.
작은 CPU fixture 통과는 이 gate의 실제 생성 증거가 아니다.

초기 설계값:

- 최종 main 순서에서 처음 등장하는 source-bound repair 지원 부모 32개를 고정한다.
- 데이터 순서는 capability가 앞부분에도 분산되도록 구성한다.
- 해당 부모의 NOTE/C가 지연 일정상 모두 닫힌 다음 판정한다.
- 서로 다른 부모에서 실제 VERIFIED_REPAIR가 최소 3개 있어야 한다.
- N의 verified-content token이 0이면 실패다.
- N의 첫 실제 성분별 backward에서 유한·비영 gradient와 저장 갱신 연결을 확인한다.
- 교정 기회 자체가 0이면 성공이 아니라 INCONCLUSIVE_SIGNAL이다.
- 3개는 효과 입증 문턱이 아니라 구조가 실제로 작동하는지 확인할 최소 설계값이다.

동결 자료에서 지원 부모가 32개 미만이면 GPU 진입 전에 SIGNAL_COVERAGE_INSUFFICIENT로 종료한다.
32번째 부모의 정확한 block과 NOTE/C carry·drain까지 필요한 slot을 CPU로 산출해 plan에 넣는다.
예를 들어 지원 159개가 대략 균등하게 섞이면 32번째는 main 약 604번째, 약 38묶음 근처다.
v16b의 144.2초/묶음을 단순 대입하면 약 1.52시간에 carry·검사 비용이 더해진다.
이는 v17 시간 예측값이 아니며 실제 고정 순서의 정확한 위치와 새 손실 비용으로 다시 계산한다.
신호 검사 전후에 main을 다시 시작하거나 첫 32개를 재생성하지 않는다.
이 gate에서는 지원 부모 32개만이 아니라 그 시점까지 닫힌 전체 main T1의 길이 분포를 함께 판정한다.
교정 N이 3개 나왔더라도 별도 붕괴 STOP이 있으면 본학습을 계속하지 않는다.
점진적 길이 WARNING은 정확도·잘림·출처 구성과 함께 보고하며 N 성공으로 숨기지 않는다.

실패 시 다음 부모를 몰래 추가해 통과율을 높이지 않는다.
다음 안전한 drain/checkpoint에서 멈추고 결과를 보존한다.
기준을 낮춰 같은 실행을 이어가지 않는다.
단, 이 gate는 실행 승인 시 함께 봉인할 신규 중단 규칙이다.
현재 실행 중인 포드에 적용하는 조치가 아니다.

### 12.3 일반 능력 감시

기존 길이·반복·NO_FINAL·IS drift guard는 유지하되 v17에 맞는 입력 계약을 검증한다.
여러 중간 확인의 거짓 정지율을 과거 자료 재표집으로 검사한다.
과거 문턱을 새 손실·새 데이터에 자동 인증하지 않는다.

현재 GuardV16B의 threshold가 reference=0일 때 recent=0에도 발동할 수 있는 경계를 검사한다.
v17 count-rule은 `minimum = max(4, ceil(2*ref), ceil(ref + 4*sqrt(ref)))`로 고정한다.
recent count가 minimum 이상일 때만 발동한다. ref=0/recent=0은 통과하며 4는 새 설계 하한이다.
유효 표본수가 달라지는 마지막 묶음에는 ref를 `recent_n/ref_n`으로 환산한다.
옛 v16b guard는 소급 변경하지 않고 v17 분기·새 false-alarm receipt로 검사한다.
관측이 없으면 null이며 정상 0으로 바꾸지 않는다.
서로 다른 훈련 문항 묶음의 정확도 차이를 통제된 학습 효과라고 보고하지 않는다.

### 12.4 점진적 단답 증가와 길이 양극화

기존 `두 묶음 연속 short≥12/16` 급격한 붕괴 규칙은 유지한다.
그 규칙으로는 완만한 변화를 잡지 못하므로 다음 신규 v17 규칙을 추가한다.

    기준: 최초 24개 main 묶음
    비교: 최근 12개 main 묶음
    검사 시작: 36번째 묶음부터, 각 완결 묶음에서 1회
    short: 실제 T1 completion_tokens < 32
    p_ref = short_ref / n_ref
    p_recent = short_recent / n_recent
    delta = p_recent - p_ref
    pooled = (short_ref + short_recent) / (n_ref + n_recent)
    se = sqrt(pooled * (1-pooled) * (1/n_ref + 1/n_recent))
    SHORT_DRIFT_WARNING: delta >= 0.05
    SHORT_DRIFT_STOP: delta >= max(0.15, 4*se)

5pp 경고는 13%→21% 같은 증가를 보고하기 위한 설계값이다.
STOP은 더 큰 변화의 보수적인 보호 규칙이며 4*se를 공식 유의확률이라고 부르지 않는다.
두 기준은 성능의 최적값이 아니고 새 protocol과 guard receipt에 고정한다.
마지막 8문항 묶음도 실제 n을 사용하고 12×16으로 분모를 채우지 않는다.
36묶음 전은 WARMUP이며, 길이 분포는 보고하지만 이 trend 판정을 만들지 않는다.

실제 v16b의 같은 창을 CPU로 재집계하면 최초24는 51/384=13.28%,
마지막12는 38/184=20.65%, 최대 recent12는 46/192=23.96%였다.
사용자가 인용한 63/312는 다른 구간이므로 last12와 혼합하지 않는다.
새 WARNING은 마지막 구간의 +7.37pp를 보고하지만, 그 자체가 정확도 하락의 인과 증거는 아니다.

short 비율 증가와 truncated 비율 증가가 동시에 나타나면 LENGTH_POLARIZATION_WARNING을 덧붙인다.
길이 중앙값 하나가 비슷해도 양쪽 꼬리가 늘어나는 현상을 숨기지 않는다.
이 경고는 토큰 상한 자동 변경·짧은 목표 자동 삭제·A1 비중 자동 변경으로 연결하지 않는다.

전체 STOP 규칙을 함께 넣어 고정 F0 자료를 최소2,000회 재표집한다.
v17의 실제 source/capability 순서·결측 위치·마지막 묶음 크기를 유지하고 seed를 고정한다.
한 번의 검사별 비율이 아니라 전체 실행 중 한 번이라도 STOP한 비율의 95% 상한≤2%를 검사한다.
이 검증이 부족하면 미검증으로 남기고 실제 자료가 없는 tail/IS 항목을 정상 0으로 채우지 않는다.
새 문턱의 값은 본학습 응답을 보고 조정하지 않는다.

외부 개발200은 F0·준비 직후·신호 gate 시점·최종에서 동일 문항의 길이 분포와 정오를 함께 본다.
gate 측정은 최종 checkpoint 선택용 여러 MMMU 평가가 아니며 외부 개발 측정이다.
이 추가200문항 생성 시간을 §18 ETA에 넣고 기존 수집과 중복 실행하지 않는다.
외부 집합의 short/잘림 변화와 main의 서로 다른 문항 창 통계를 따로 보고한다.

## 13. 모델 선정과 비교 정책

최종 모델을 선택하는 것은 허용된 사용자 결정이다.
문항마다 여러 답 중 좋은 답을 고르는 것과 구별한다.

v17 기본안:

- F0와 v17의 독립 외부 효과 비교를 기본으로 한다.
- 최종 개발용 200문항에서 같은 F0 대비 회복·손실을 계산한다.
- 순변화가 음수이고 정확한 양측 McNemar p<0.05이면 DEGRADED로 판정해 v17의 자동 MMMU 진입을 막는다.
- 위 조건에 해당하지 않아도, 개발용 200과 사전 고정 효과 집합 모두 순변화≤0이면 EVAL_REVIEW_REQUIRED다.
- EVAL_REVIEW_REQUIRED에서는 두 집합의 회복·손실·길이 분포·남은 예상 시간을 보고하고 사용자 결정 후 900 평가에 진입한다.
- 이 보고와 최종 MMMU 보고는 모두 §5.4의 원본 종료 상태별 분해를 함께 낸다.
- 효과 집합이 미완료·공급 부족·노출 불명확이면 순변화 0으로 채우지 않고 같은 검토 상태로 남긴다.
- 둘 중 하나만 양수여도 성능 개선 인증은 아니다. 다른 무결성·붕괴 gate를 통과한 예정 최종 모델 하나만 평가 후보가 된다.
- 이 개발용 규칙은 비용 보호를 위한 사전 설계이며 65점 예측기나 독립 확증 검정이 아니다.
- 중간 checkpoint를 여러 개 생성해 MMMU 점수로 탐색하지 않는다.
- 이번 학습의 예정 최종 checkpoint 하나를 MMMU 평가 대상으로 등록한다.
- 안전 중단 checkpoint를 자동 최종 모델로 승격하지 않는다.
- 중단 시 해당 checkpoint 평가 여부는 별도 판단이다.
- 과거 v15를 대안으로 비교할 때는 그 모델의 학습 노출을 고려한다.

이 문서는 “v15 이상 성능”을 출발 가정으로 두지 않는다.
v15의 전체 900 결과는 없으므로 280/450을 두 배로 환산하지 않는다.

## 14. 일정·정책 버전·재개를 보존하는 구현

### 14.1 기존 일정 유지

V16bScheduler는 이미 다음 T1/OBS와 지연 NOTE/RESOLVE를 함께 넣는다.
이 일정을 v17에서도 재사용한다.

- 두 활성 정책 버전 상한.
- η=1.
- note carry 1.
- resolve carry 3.
- 고정 부모 순서.
- 기존 prefill barrier와 update boundary.

옛 요청이 시각 가중치 교체 뒤 선점 재계산되면 LoRA reference가 남아 있어도 mixed_version이 될 수 있다.
SingleVllmSampler.step의 기존 검출을 보존하고 §10.5의 1건 HOLD 규칙으로 연결한다.
reference count만으로 응답 전체 정책 불변을 증명했다고 하지 않는다.

HF backward와 vLLM decode를 같은 GPU에서 동시에 실행하지 않는다.
더 많은 정책 버전이나 여러 GPU를 이번 기본안에 추가하지 않는다.

### 14.2 A1과 N/C의 소비 위치

A1은 해당 T1의 기존 RL slot에 들어간다.
N과 C는 기존 지연 일정이 지정한 slot에 들어간다.
N이 수용되었다는 이유로 C를 기다리거나 반대로 C 정답 때까지 N을 보류하지 않는다.
행 ID·부모 ID·policy version·source generation hash를 별도로 소비 원장에 기록한다.

### 14.3 CPU 병렬화가 바꾸면 안 되는 것

- 요청의 admission version.
- baseline history cutoff.
- 원래 요청 seed.
- update가 소비하는 부모와 행 목록.
- scheduler가 진행하는 slot 수.
- Vision/Merger prefill 시점의 정책.
- checkpoint가 닫는 journal prefix.

worker 결과 도착 순서로 학습 순서를 바꾸지 않는다.
시간이 빨라져도 인과적으로 앞선 결과만 baseline에 사용할 수 있다.
작업 결과를 봉인된 slot 순서로 반영한다.

### 14.4 별도 호스트 대기열은 기본안에서 제외

호스트에 요청을 보류하면 아직 엔진에 들어가지 않은 요청도 옛 정책을 참조한다.
그 요청을 refcount·prefill barrier에서 빠뜨리면 이미지 처리 전에 가중치가 바뀐다.
따라서 단순한 host queue 추가를 성능 개선으로 넣지 않는다.

필요한 대기열은 기존 엔진 경로를 사용한다.
추가 queue를 구현하려면 queued/prefilled/running/completed 상태와 drain/refcount 시험을 별도로 완성해야 한다.

### 14.4.1 학습 KV 사전 검사와 제출 속도 제어

학습용 엔진과 평가용 엔진의 KV 예산은 다르므로 검사 영수증을 공유하지 않는다.
회수본 학습 KV는 174,752token이었다. 실제 생성4,953건에서 preempted=0·mixed_version=0을 확인했다.
이는 당시 실행의 관측이며 새 A1/F0 계산·새 입력 길이에 대한 용량 인증은 아니다.
32칸에서16칸으로 줄였다는 사실만으로 최악 길이의 안전을 보장하지 않는다.

    train_max_num_seqs = 16                 # v17의 동시 요청 상한 설계값
    usable = floor(0.90 * actual_train_KV_tokens)
    reserve(request) = ceil((actual_prompt_tokens + max_new_tokens)
                            / block_size) * block_size
    sum(reserve(r) for all admitted running AND waiting r) <= usable

실제 prompt 길이는 모델에 보낼 이미지·chat template 처리가 끝난 token 길이다.
준비 전 길이를 모르면 해당 단계의 인증된 보수적 문맥 상한을 예약한다.
예약을 현재 생성 길이만큼만 잡아 미래 decode 용량을 과소평가하지 않는다.
request 하나가 usable보다 크면 첫 제출 전에 CAPACITY_INSUFFICIENT로 종료한다.

역사적 예에서 16×24,576=393,216은 KV174,752보다 크다.
10% 여유를 둔 최악 문맥 요청 상한은 floor(0.9×174,752/24,576)=6이다.
짧은 OBS·NOTE·T1은 실제 최대 요청 길이로 예약하므로 항상6개만 실행한다는 뜻은 아니다.
16은 목표로 강제 채워야 하는 수가 아니라 예약 검사 위에 놓인 상한이다.

최소 구현은 `V16bScheduler._submit_stage`의 동기식 제출 속도 제어다.
예약이 부족하면 이미 있는 `runtime.step()`을 진행해 완료·검증·예약 해제를 처리하고 다시 확인한다.
같은 제출 순서·정책판·seed·미리 봉인한 baseline cutoff를 유지한다.
`submit_for_version`의 모든 예정 제출/skip 결정이 끝나기 전에는 optimizer update·새 정책 push를 하지 않는다.
미제출 request를 완료로 표시하거나 checkpoint의 drained 상태에 숨기지 않는다.
별도 비동기 host queue나 worker가 요청을 나중 정책으로 제출하는 경로는 만들지 않는다.
기다리는 동안 live 요청이0인데 예약을 확보할 수 없으면 무한 대기 대신 용량/원장 오류로 종료한다.
미제출 stage cursor는 원래 상태로 남기고 안전 중단에서도 제출 완료를 꾸며 쓰지 않는다.

SingleVllmSampler.submit은 이미지 처리·입력 hash를 보존한 준비 결과와 예약량을 받도록 연결한다.
같은 입력을 예약용과 실행용으로 다르게 전처리하지 않는다.
완료·명시적 abort/실패에서 예약을 정확히 한 번 해제하고 native waiting도 refcount에 남긴다.
prefill_barrier·옛 정책 reference·두 판 상한은 유지한다.
사전 예약은 선점 억제책이며 모든 메모리 장애를 불가능하게 만든다는 증명은 아니다.

heartbeat에는 train_KV_capacity/reserved_tokens/active/waiting/admission_wait_seconds/preemption_delta를 추가한다.
선점은 즉시 경고하고, 실제 old-vision 재계산으로 mixed 판정된 응답은 기존1건 공동 HOLD 규칙을 적용한다.
선점1건과 정책 혼합1건을 같은 사건이라고 보고하지 않는다.
학습 상한16·예약 검사는 새 실행 전에 고정하며 실행 중32/48로 자동 확대하지 않는다.
v16b의16/32/48 시간 동률 관측은 참고값이고 새 예약 방식에서도 같다는 보장은 아니다.
새 공동 손실의 최대 FULL backward 메모리를 측정한 뒤 남는 실제 KV를 다시 읽는다.

### 14.5 재개 상태

다음 상태를 같은 checkpoint에 묶는다.

- optimizer, scheduler, RNG.
- global slot와 optimizer update 수.
- consumed parents와 모든 row IDs.
- 생성 정책 버전·hash.
- baseline 이력·cutoff.
- guard와 신호 gate의 분모·판정 상태.
- 닫힌 generation journal.
- drain state와 보존된 완료 출력.
- 데이터·protocol·source·입력 manifest.

열린 KV cache를 다른 엔진에서 이어받았다고 주장하지 않는다.
저장은 기존처럼 요청을 닫고 무결성이 확인된 경계에서 수행한다.

## 15. 평가 KV·시간 정체 문제의 직접 수정

### 15.1 기존 auto_slots 사용 중단 범위

기존 auto_slots는 평균 문맥 12,000을 가정한다.
v17 B2 평가에는 이 auto 선택을 사용하지 않는다.
과거 버전의 auto 결과는 바꾸지 않는다.

v17 기본 동시성 설계값은 8이다.
새 엔진의 실제 KV 용량을 읽어 첫 요청 전에 검사한다.

    reserved_per_request =
        ceil(max_model_len / kv_block_size) * kv_block_size

    required = configured_concurrency * reserved_per_request
    usable = floor(0.90 * actual_kv_token_capacity)

    require required <= usable

actual_kv_token_capacity는 예상 GPU VRAM 공식으로 대신하지 않는다.
고정된 vLLM 버전의 실제 KV block 수와 block size에서 얻는다.
속성·구조·버전이 다르면 사전 검사 실패다.

### 15.2 역사적 수치로 한 계산 예

- 당시 KV: 444,416token.
- max_model_len: 40,960.
- 10% 여유를 둔 수용 상한: floor(0.9×444,416/40,960)=9.
- 동시 8은 이 역사적 조건에서 통과한다.

이는 새 A100 실행의 실제 용량 보장이 아니다.
모델 배치·메모리 사용이 달라지면 다시 검사해야 한다.
부족할 때 상한·이미지·정밀도를 몰래 줄이지 않는다.
첫 질문을 넣기 전에 실패하므로 평가 답을 다시 뽑는 상황을 만들지 않는다.

### 15.3 어디를 수정하는가

proof_eval의 다음 연결을 함께 수정한다.

- resolve_slots.
- variant_registration.
- registered_variants.
- widened backend/LLM 생성 경로.
- lora_llm 및 preflight.
- plan의 evaluate argv.

신규 v17 실행 registration에 slots=8과 capacity 검사 규칙 hash를 기록한다.
현재 registered_variants.init의 on_backend는 실제 엔진 생성 전 호출되므로 실제 KV 측정치를 그때 만들지 않는다.
엔진·LoRA 설정이 끝난 뒤 첫 add_request 전에 capacity를 검사하고 invocation 영수증에 등록 hash와 연결한다.
재개 시 실제 KV 용량은 달라질 수 있으므로 새 측정치를 append-only 영수증으로 남긴다.
같은8칸의 고정 수용 조건을 다시 만족해야 하며 기존 registration을 새 값으로 덮어쓰지 않는다.
기존 src/mmdl/evaluation/backends/vllm_backend.py를 고치지 않는다.
기존 fine_tuning 안의 adapter 범위를 벗어나는 전역 monkey patch를 만들지 않는다.
변경한 생성기 hook은 context 종료 때 반드시 복구한다.

### 15.4 P0·B2와 실행 조건

P0·sampling·최대 출력·이미지 budget·dtype·V8은 B2 의미 조건을 유지한다.
GPU와 동시 처리 실행 조건은 baseline과 다를 수 있으므로 별도 execution identity로 기록한다.
동시성을 바꿔도 출력이 비트 단위로 같다고 주장하지 않는다.
평가 중 slots를 변경하며 같은 run으로 재샘플링하지 않는다.
프로세스가 중단돼도 같은 등록의 미완료 ID만 재개하는 것은 §15.10의 조건으로 허용한다.

**최종 평가는 합치지 않은 조정층으로 서빙한다(비병합).**
v14·v15에서 학습한 조정층을 BF16 원본 가중치에 합치면 변화의 24~55%가 반올림으로 사라짐을 실측했다
(v14 언어 조정층은 45~47%만 남음). 그래서 v15부터 평가는 vLLM 조정층 + 시각 hook의 비병합 경로를 쓴다.
- export는 `unmerged-lora` 종류만 허용하고, 병합 가중치 export를 평가 입력으로 쓰지 않는다.
- registration에 서빙 종류를 기록하고, 병합 export가 들어오면 첫 요청 전에 거절한다.
- 평가 전 PF5(학습 후 비영 조정층)의 HF↔vLLM 일치와 효과 기울기 검사를 통과해야 한다(§17.1).
- 단순화를 이유로 병합 서빙으로 되돌리지 않는다.

처음에는 기존 ID 순서를 유지한다.
문항 해시 섞기를 KV 안전의 대체물로 넣지 않는다.
새 순서를 선택한다면 실행 전에 별도 고정해야 하며 답을 본 뒤 재배열하지 않는다.

### 15.5 heartbeat

학습 SingleVllmSampler.step과 평가 engine.step 관측 경로에서 기록한다.

    wall_time
    engine_steps
    requests_running / waiting / completed
    new_output_tokens_since_last
    max_observed_output_tokens_by_request
    prefill_tokens_since_last
    kv_blocks_used / total
    cumulative_preemptions
    last_new_token_at
    last_compute_progress_at
    oldest_request_seconds

설계상 기록 간격은 10초다.
매 token마다 전체 원문·tensor를 직렬화하지 않는다.
마지막 상태 파일은 atomic replace하고 사건 로그는 append-only로 둔다.

### 15.6 정체와 긴 정상 응답을 구별

초기 경고 설계값:

- 180초 새 출력 token 없음: TOKEN_STALL_WARNING.
- 같은 구간 preemption 증가: KV_RECOMPUTE_WARNING.
- token과 prefill/계산 모두 진행 없음: ENGINE_NO_PROGRESS_WARNING.

이 수치는 성능 보장값이나 자동 종료 문턱이 아니다.
완료 문항 0개여도 새 token이 생성되면 정상 긴 응답일 수 있다.
engine.step 횟수만 늘면 재계산 정체를 놓칠 수 있다.

경고는 예외를 던지거나 iterator를 닫지 않는다.
기존 generate_stream은 iterator 종료 시 활성 요청을 abort한다.
경고가 자동 재시작·재샘플링·답 폐기로 이어지면 안 된다.
승인된 자금·전체 시간 종료는 별도 경로이며 INCOMPLETE를 보존한다.

### 15.7 vLLM 통계

평가 LLM 생성에도 disable_log_stats=False를 명시한다.
학습 경로에는 이미 이 설정이 있다.
stdout 통계와 자체 구조화 heartbeat를 함께 보존한다.
ptrace 가능 여부에 모니터링을 의존하지 않는다.

### 15.8 8칸 연속 처리 — 하나가 끝나면 즉시 다음 문항

기존 `VLLMBackend.generate_stream`의 fill_slots를 재사용한다.
이미 최대 동시 요청 수를 채우고, 응답 하나를 반환한 뒤 비워진 자리에 다음 문항을 넣는다.
8개를 한 묶음으로 기다렸다가 다음8개를 시작하는 방식으로 바꾸지 않는다.

    pending = original_900_ids - durable_completed_ids
    처음 min(8, len(pending))개 제출
    while active:
        engine.step()
        완료 응답마다:
            원래 ID/seed/입력/정책 binding 검사
            최종 raw 영수증과 샘플을 영구 확정
            같은 원래 순서의 다음 미완료 ID를 빈칸에 제출

예를 들어 문항1이32K까지 가고 문항2가 먼저 끝나면 문항1을 기다리지 않고 문항9를 시작한다.
동시 active는8을 넘지 않으며 각 문항의 답은 독립된 단일 응답이다.
active8개 중 정답 후보를 골라 쓰거나 여러 답을 합치는 방식이 아니다.
고정8칸은 KV 선점 위험을 줄이는 값이며,32칸보다 항상 느리거나 항상 빠르다는 실측 결론은 없다.
선점·재계산을 줄이면 실제 완료 처리량은 오히려 좋아질 수 있다.
원문 저장·fsync 비용도 처리량에 포함하고 내구성을 없애서 빠른 숫자를 만들지 않는다.

### 15.9 실측 기반 평가 시간과 watchdog

v16b의 plan._check_v16b에는 `timeouts.evaluate == 23400` 검사가 있다.
proof_eval의 기존900 사전등록도 evaluation_watchdog_seconds=23400을 기록한다.
v17에는 별도 plan 검사·사전등록 버전을 추가해 실측 영수증에서 유한한 timeout을 계산한다.
옛 v16b의6.5시간 계약·hash는 변경하지 않는다.
새 값은 plan 단계 timeout, 사전등록, runner watchdog, 자금 예약 계산에 동일하게 연결한다.

짧은 실제 확인은 승인된 장비에서 수행한다.
MMMU 문항을 속도 시험으로 다시 풀지 않는다.
본학습·prep·개발·효과 집합과 겹치지 않는 비벤치마크 performance-only 자료를 최대16개 사전 고정한다.
performance-only 출력은 학습·효과 점수·최종 답 선택에 사용하지 않는다.
같은 BF16 backend·8칸·이미지 정책·B2 sampling에서 연속 처리를 측정한다.
초기 성능 probe는 최대10분이며 미완료 출력은 진단용 부분 응답으로만 보존한다.
이 probe의 시간 상한을 실제 MMMU 답을 자르는 규칙으로 재사용하지 않는다.

측정 항목:

    엔진 로드·adapter 적용 시간
    실제 입력 token·이미지 patch 수
    engine wall seconds와 생성 token 증가량
    active/waiting 분포, prefill·decode·tail 시간
    KV 여유·선점 횟수, 저장·finalize 시간
    관측된 context 길이 범위와 긴 응답 coverage

샘플별 latency를 합산해 총 처리량 분모로 쓰지 않는다.
q_run은 전체 probe의 실제 평가 wall time 대비 생성 token 수다.
60초 창의 token/wall 비율 중앙값과 q_run 중 작은 양수를 q_safe로 쓴다.
0token·짧은 context만 관측·선점 발생 등으로 대표성이 부족하면 CALIBRATION_LOW_COVERAGE로 표시한다.
그때 짧은 문제의 빠른 속도를32K tail까지 확인한 것으로 쓰지 않고 시나리오 범위를 보고한다.
실제 본평가에서는 동일900 실행의 앞부분을 계속 사용해 ETA를 갱신하며 별도 예비 MMMU 실행을 만들지 않는다.

초기 예산의 명시적인 계산 예:

    historical_output_tokens = 4_116_922
    stress_output_tokens = 1.5 * historical_output_tokens
    estimated_work = stress_output_tokens / q_safe
    eval_watchdog_seconds = ceil(1.25 *
        (estimated_work + measured_load_finalize_allowance + declared_long_tail_allowance))

1.5와1.25는 초기 시나리오 여유계수이며 통계적 상한·속도 실측값은 아니다.
long_tail_allowance는 단독으로 남은 긴 요청의 속도 관측/추정과 범위를 영수증에 명시한다.
측정이 충분하지 않으면 timeout을 임의 PASS로 봉인하지 않고 예상 범위와 근거 부족을 보고한다.
최종 checkpoint의 실제 평가 길이가 더 길면 이 계산을 넘을 수 있다.

평가중 ETA는 완료된 짧은 문항 평균만 사용하지 않는다.
활성 요청의 생성 길이·남은 cap, 아직 안 들어간 ID 수, 누적 tokens와 선점 상태를 함께 기록한다.
정오 점수는 timeout 산정·미완료 ID 선택에 사용하지 않는다.
진행 ETA를 갱신해도 동결된 watchdog을 몰래 연장하지 않는다.
연장이 필요하면 현재 상태와 남은 시간·비용을 보고하고 사용자 결정에 따른다.

실제 funding 종료 경계가 있다면 `min(평가 watchdog 종료, funding 종료−회수 예약)`를 쓴다.
기존 funding_decision의 post_train_hours=5.65를 새 실측 평가 예약으로 교체하는 v17 분기를 둔다.
14~17시간 목표 자체는 funding 종료 경계가 아니다.
timeout 전 경고·상태 저장을 수행하고, 만료 시 이미 완료한 응답은 모두 보존한다.

### 15.10 같은 등록에서 미완료 문항만 재개

기존 engine.run은 pending을 expected_ids−writer.completed로 계산한다.
RunWriter의 원자적 no-overwrite 저장·샘플 hash 검사·독점 lock·resume 검사를 재사용한다.
중단 시 원문이 전혀 안 남는 것이 아니라 부분 샘플이 남고900 점수가 완성되지 않는 것이 현재 문제다.

동일 logical run에서 고정할 것:

    preregistration hash, START ledger entry, run_id
    checkpoint/model/processor/source/protocol hashes
    original 900 expected IDs와 순서
    sample seed 규칙, P0·B2·V8, slots=8
    hardware/environment identity와 허용 경로 계약

별도 프로세스 실행은 새 invocation_id로 기록하되 위 run_id와 등록은 바꾸지 않는다.
재개 때 ledger-start를 다시 쓰거나 ledger-fail로 기존 START를 무효화하지 않는다.
proof_eval의 single_attempt 보호를 삭제하지 않는다.
기존 pod.chain.run은 실행 폴더 재사용을 거절하므로 전체 chain을 처음부터 다시 돌리지 않는다.
`v17_eval resume`라는 평가 전용 진입점에서 등록 검사→부분 상태 검사→같은 run-eval --resume를 호출한다.
준비·학습·내보내기·사전등록을 다시 실행하지 않는다.
이미 완료된 run의 빠른 반환에서도 backend 생성 전에 등록·모델·ID 일치를 먼저 검사한다.

완료 상태의 기준은 유효한 원자 저장 sample이다.
정답·오답·NO_PARSE·length 종료도 정상적으로 확정된 평가 응답이면 완료이며 재생성하지 않는다.
원본 응답의 내용이나 정오를 보고 재개 대상을 고르지 않는다.
같은 run·같은 등록의 미완료 ID만 원래 순서로 제출한다.
v16b의 서로 다른 불완전 실행59개와9개를 이 규칙으로 소급 합치지 않는다.

중단 경계별 처리:

| 디스크 상태 | 재개 동작 |
|---|---|
| sample 확정, invocation FINISHED/완료 목록만 누락 | sample hash 검증 후 영수증만 복구. 생성0회 |
| FINAL_RAW 영수증 확정, sample 미확정 | 같은 raw/token으로 V8·record 구성·RunWriter.save 수행. 생성0회 |
| STARTED만 있고 최종 raw/sample 없음 | 미완료로 보존하고 같은 seed로 처음부터 생성 가능 |
| 모든900 sample 확정, summary만 누락 | CPU finalize·재채점만 수행. 모델 로드0회 |
| 동일 ID의 상충하는 raw/sample | 자동 선택하지 않고 무결성 오류로 중단 |

FINAL_RAW 영수증은 fine_tuning의 제한된 backend hook에서 engine 최종 출력 직후, caller에 yield하기 전에 원자 저장한다.
원래 request/input/seed/token/model/registration hash와 최종 raw를 함께 봉인한다.
복구 변환기는 기존 평가 record와 동일한 계약을 구현하고 기존 V8·RunWriter.save를 호출한다.
정상 경로와 복구 경로의 record byte/채점 동등성을 CPU 검사로 확인한다.
원문이 없는데 알고 있던 정답을 넣어 sample을 복구하지 않는다.

현재 backend는 진행 중 KV와 요청별 RNG를 디스크에 저장하지 않는다.
하드 크래시 뒤 미완료 응답을 정확한 token 위치에서 이어 쓴다고 주장하지 않는다.
동일 seed 재생성도 중단 전 부분 출력과 비트 단위로 같다는 보장은 없다.
이는 미완료 요청의 기술적 재시작이며 완료 답 여러 개 중 고르는 것이 아니다.
엔진이 종료 응답을 만들었으나 FINAL_RAW 영구 저장 전에 하드 크래시하면 그 답은 복원 불가할 수 있다.
이 창의 한계를 숨기지 않고 원문·영수증이 없는 완료를 가정하지 않는다.

논리 상태는 REGISTERED → RUNNING → INTERRUPTED_RESUMABLE → RUNNING → FULL900_VERIFIED로 기록한다.
INVALID_IDENTITY·CORRUPT_ARTIFACT는 정상 재개 상태가 아니다.
process exit 실패와 전체 logical run 무효화를 구별한다.
클라우드 재시작·시간 연장은 기존 사용자 실행 지시 범위에서만 하며 충전 예정만으로 자동 재개하지 않는다.

### 15.11 포드 장애·회수·최종 집계

기본 재개는 동일 포드에서 동일 identity를 유지하는 경로다.
새 포드는 원장·등록·모델 참조 경로·완료 샘플을 복원하고 identity 일치를 CPU로 확인한 경우만 허용한다.
동일 GPU 이름·lock·commit만으로 같은 환경이라고 판단하지 않는다.
현재 환경 hash는 Python 전체 버전, kernel release, 모든 설치 패키지, CUDA runtime,
GPU driver/총메모리/compute capability 등을 포함한다.
모델 참조 절대경로도 사전등록 계약에 포함된다.
새 환경에 맞추려고 옛 manifest를 덮어쓰거나 identity 검사를 우회하지 않는다.
일치하지 않는 새 포드로의 자동 migration은 이 기본안에서 보장하지 않는다.

닫힌 sample·FINAL_RAW·등록·원장은32개 완료 또는5분마다 증분 회수 manifest를 만들도록 설계한다.
이미 승인된 로컬/영속 수신지로 전달한 뒤 hash 검증 receipt를 보존한다.
노트북 절전으로 로컬 수신이 끊기면 그 상태를 알리고 원격 파일을 삭제하지 않는다.
유일한 사본이 장애 포드의 임시 디스크에만 있으면 포드 소실 후 재개를 보장할 수 없다.
추가 유료 저장소를 자동 생성하지 않으며 영속 대상과 비용은 실행 계획에서 정한다.

기존 collect.pack_eval 목록은 분석용으로 samples/·inference_invocations/·입력 자산 등을 제외한다.
따라서 v17 평가 재개 archive는 이를 포함한 별도 필수 파일 목록을 검사한다.
등록·원장·run/backend/config/code identity·원자 sample·raw final·입력 이미지/참조를 보존한다.
추론용 archive, 학습 재개용 archive, 평가 재개용 archive의 인증 범위를 구분한다.

현재 proof_eval은 --mode partial로900 ID를 지정하므로 summary의 PARTIAL 문자열만으로 미완료라고 판단하지 않는다.
원래900 ID가 정확히 한 번씩 존재하고 모델·입력·V8 재채점이 일치할 때 FULL900_VERIFIED로 보고한다.
여러 invocation의 시간·재시작·복구 비용을 모두 합산하고, 서로 다른 run의 결과는 합치지 않는다.
Compass 보완 판정은 동일 logical run의900개 원문이 완성된 뒤에 수행한다.

## 16. 학습 시간 최적화

### 16.1 측정부터 분리

매 slot에 다음 wall time을 기록한다.

- 요청 준비·이미지 decode/processor.
- 생성 대기·engine step.
- 채점.
- OBS/NOTE 검증.
- 행 생성·token mask.
- 현재 policy forward/backward.
- F0 reference forward와 KL projection.
- optimizer step.
- LoRA/Merger 전송·판 일치 검사.
- checkpoint·probe·drain.
- 파일 해시·직렬화.

host wall time을 GPU 연산 시간이라고 부르지 않는다.
GPU utilization·전력 표본에는 기간·간격·표본 수를 붙인다.

### 16.2 현재 기록으로 가능한 결론

v16b runtime의 191 slot host timing 합계는 약 7.308시간이다.
구성은 생성 대기 약 3.458, 갱신 2.780, checkpoint 0.558, 기타 0.512시간이다.
겹친 요청 수명과 측정 구간의 정의를 고려해야 하며 GPU kernel별 프로파일은 아니다.
이 191개는 본학습 188 + 마무리 3이다. 준비 SFT 7·설치·평가·회수는 제외한다.
drain_generation_seconds 0.156719시간은 generation의 부분집합이므로 다시 더하지 않는다.

기타 overhead를 모두 없애도 약 7% 수준이다.
CPU 해시 개선만으로 전체 시간을 절반으로 줄인다고 약속할 수 없다.
v17은 A1과 추가 F0 계산 때문에 갱신 시간이 증가할 수 있다.
현재 일정에서 긴 응답을 기다리는 장벽을 제거한 것이 아니므로 전체 학습이 v16b보다 빨라진다고 약속하지 않는다.
3.458시간은 sampler 진행과 출력 처리의 host 시간이며 전부 GPU 유휴·낭비 시간이 아니다.
별도 GPU나 정책 지연 확대 없이 확보할 시간은 계산 공유·불필요한 I/O 감소·KV 선점 억제에서 측정한다.

### 16.3 우선 적용할 낮은 위험 변경

1. 이미 버전당 한 번 계산하는 policy hash를 요청마다 다시 계산하지 않는 현재 구현 유지.
2. 같은 불변 이미지의 검증 결과·입력 준비 cache 재사용.
3. 큰 파일 hash는 기존 streaming file_digest helper 사용.
4. 대형 tensor를 .tobytes로 복제하는 비용이 측정되면 같은 byte 순서의 chunk/memoryview 검토.
5. 같은 문제의 N/C 검증 receipt를 다시 계산할 때 canonical 결과를 비교하되 불필요한 I/O 반복 제거.
6. 비동기 해시에 live GPU tensor를 넘기지 않음.
7. 작은 준비·SFT 출력의 불필요한 전체 snapshot 저장 제거는 resume 필수 파일과 구분.

cache key에는 이미지·prompt·processor revision·image policy가 들어간다.
basename만 같은 이미지끼리 cache를 공유하지 않는다.
mtime만으로 무결성을 인증하지 않는다.

### 16.3.1 같은 T1의 RL+A1 forward를 공유

A1 복원 때문에 동일한 긴 T1을 두 번 forward/backward하는 구현은 기본안에서 피한다.
학습 행·receipt·분모는 분리한 채 계산만 공유한다.

공유 key는 `policy_sha256 + training_input_sha256 + route + sampled_action_ids_sha256`다.
key가 모두 같고 같은 공동 update에 있는 RL/A1만 묶는다.
각 vocabulary chunk에서 같은 raw logits로 raw log p와 processed log q를 만든다.
그 chunk의 A1 CE와 RL 항을 합쳐 한 그래프에서 backward한다.
가능하면 같은 current hidden의 선택 위치를 F0 KL에도 사용한다.
F0 reference forward는 별도로 필요하며, 이것까지 없어진다고 보고하지 않는다.

PG와 A1의 token mask·행 분모·수용 판정은 각각 유지한다.
RL reward가 0이거나 PG가 제외돼도 A1의 자체 수용 조건이 유효한지 별도로 검사한다.
단, mixed-policy/HOLD는 공유 계산 전체를 버린다.
C는 lesson 없는 학습 입력으로 바뀌므로 생성 당시 입력 key로 공유하지 않는다.
N과 다른 prompt의 내용을 같은 hidden이라고 가정하지 않는다.

분리 계산을 기준 oracle로 쓰는 작은 tensor 검사에서 loss와 모든 학습 gradient를 비교한다.
공유 경로를 성능 최적화라고 승인하기 전에 수치 동등성과 최대 길이 메모리를 확인한다.
추가 생성 없이 A1을 쓰는 것은 확실하지만 추가 학습 계산이 완전히 0이라는 뜻은 아니다.

### 16.3.2 A1·64위치 F0 보존 비용을 실제로 분리 측정

승인된 BF16 장비의 짧은 사전 확인에서 동일한 봉인된 입력·sampled action·mask로 다음 계산을 비교한다.

    A = v17 PG 계산
    B = 같은 PG + 공유 forward A1
    C = 같은 PG + 공유 A1 + 실제 64위치 F0 KL
    delta_A1 = time(B) - time(A)
    delta_F0 = time(C) - time(B)

이는 시간·메모리 probe이며 별도 학습 모델3개나 추가 답 후보를 만드는 실험이 아니다.
optimizer.step 없이 gradient를 지우고 가중치·RNG·모드·route를 복구한다.
warmup1회 뒤 같은 계약으로 측정3회, 구간 앞뒤 CUDA 동기화, 중앙값과 범위를 기록한다.
짧은/중간/긴 실제 학습 자료 행과 최대8K FULL 용량 probe를 포함한다.
크기 맞춤 capacity fixture는 실제 풀이·교정·정답률 증거로 사용하지 않는다.

F0 전체 prefix forward, 선택 위치의 vocabulary KL, current forward/backward,
peak allocated/reserved를 따로 기록한다. 64위치 projection만의 비용을 전체 F0 비용으로 쓰지 않는다.
FULL은 이미지 수·patch 수·prompt 길이도 묶어 기록한다.
이미지 없는 LM 경로를 이미지 FULL 경로 시간의 대체물로 쓰지 않는다.
기존 v16b 단일 RL 메모리 probe만 통과하고 v17 공유 손실 용량이 확인됐다고 하지 않는다.

실제 수용 A1/C/N 수·길이 분포로 가중한 증가분을 main188묶음과 마무리3슬롯 ETA에 넣는다.
진단 반복의 시간도 preflight에 포함하고, 기본 PG 시간을 두 번 더하지 않는다.
초기4묶음의 실제 공동 갱신 시간과 비교해 추정치를 갱신한다.
`+10~30%`는 검토자가 제시한 시나리오일 뿐 실측값으로 봉인하지 않는다.
최대 행이 너무 비싸도 KL을 자동으로 끄거나 기존4위치로 돌아가지 않는다.

### 16.4 CPU worker는 조건부 구현

첫 계측에서 순수 CPU 단계가 전체 wall time의 5% 이상을 차지할 때만 적용한다.
이 5%는 구현 우선순위 설계값이며 실행 중 알고리즘 변경 기준이 아니다.
본학습 시작 전 검증·동결해야 한다.

첫 구현은 spawn 방식 worker 2개로 제한한다.
순수 파일·decode·검증 입력만 전달한다.
살아 있는 GPU 모델, 엔진, optimizer, 공유 processor 상태를 전달하지 않는다.
worker는 CUDA를 초기화하지 않는다.
반환값은 요청 ID·입력 hash·검증기 hash와 묶는다.
부모 프로세스가 원래 순서로 receipt를 봉인한다.
worker 수 1/2에서 결과 byte·요청 순서·baseline cutoff가 같아야 한다.

### 16.5 추가 병렬 학습은 제외

본학습 묶음 수·effective batch·update 순서가 바뀌는 큰 병렬화는 기본안에 넣지 않는다.
GRPO 4회, 다중 GPU, 비동기 optimizer, 더 큰 policy lag는 이번 문제 해결의 필수 조건이 아니다.
실행시간을 줄이기 위해 학습 원리를 동시에 바꾸면 원인 분리가 더 어려워진다.

## 17. PF5·doctor·로컬 검사 공백

### 17.1 F0와 학습 후 검사 분리

| 상태 | 필요한 검사 | 해석 |
|---|---|---|
| F0의 LoRA B=0 | base 불변·슬롯·hook·HF↔vLLM 일치 | 효과 검정력 없음이 정상일 수 있음 |
| 준비 후 비영 adapter | 위 검사 + 효과 기울기·상관 | 실제 적용 효과 확인 |
| 최종 checkpoint | 파일·전송·HF↔vLLM·효과·reload | 평가 가중치 적용 검증 |

F0에서 power=false를 관측했다고 실패한 효과 검사를 PASS로 바꾸지 않는다.
해당 단계에서 효과 존재를 요구하지 않는 별도 계약으로 기록한다.
학습 후에도 효과가 미측정이면 그 사실을 남긴다.
기울기·상관 문턱을 낮춰 통과시키지 않는다.

### 17.2 doctor

plan에 명시적 doctor 단계를 prep-generate보다 앞에 넣는다.
현재 환경 CLI의 필수 --output을 포함한다.
expect_files로 출력 존재를 확인하고 내용의 CUDA/BF16·디스크 검사도 판정한다.
shell의 성공 코드만으로 정상 GPU라고 하지 않는다.

### 17.3 로컬 검사 범위

- 로컬은 기존 NF4 cache로 짧은 기능 검사만 한다.
- 새 7B 모델을 다운로드하지 않는다.
- 로컬 8GB에서 BF16 4B vLLM 전 경로가 가능하다고 요구하지 않는다.
- tiny fixture는 CPU·vLLM 연결 계약 검사용이며 실제 모델 효과와 구별한다.
- 실제 로컬 smoke는 최대 8부모·4 main 묶음 수준을 기본으로 한다.
- 자동 장기 L0·수백 문항 생성·장시간 탐색을 시작하지 않는다.
- GPU 실행 자체는 별도 사용자 지시 범위에서 한다.

실제 BF16 vLLM 경로의 짧은 확인은 승인된 장비에서 본학습 전에 한다.
이 검사를 로컬 fixture 통과로 대체하지 않는다.

### 17.4 실행 전 확인 목록

- main3000·prep512·개발200 역할과 최종 ChartQA 연결 후 OBS/N 공급 보고.
- 전체 STOP 규칙의 false-alarm 영수증과 신호 gate 위치.
- 새 A1/F0 공동 손실의 BF16 메모리·시간 영수증, 학습 KV 예약 검사.
-8칸 연속 평가 처리량·coverage·계산된 watchdog과 동일 등록 재개 CPU 검사.
- 신규15시간 하드 차단이 없고3,000개 전체를 소모하는 일정인지 확인.
- 실제 충전 반영·가격·평가/회수 예약금 확인. 과거16달러를 고정 상한으로 사용하지 않음.
- 샘플 증분 회수 대상·identity·장애 시 복구 가능 범위 확인.

시간·금액은 실행 당시 측정·확인해 보고하며, 이번 문서 보강 중 원격 확인을 했다고 기록하지 않는다.

## 18. 감시·비용·회수

### 18.1 감시는 실행 장비에서 지속

기존 runner·start·watchdog을 재사용한다.
규칙 감시·heartbeat·자금 종료를 게시된 포드 코드에 포함한다.
노트북 덮개·로컬 SSH 세션·로컬 폴링이 끊겨도 작동해야 한다.

PID·startticks·명령·cwd로 소유 프로세스를 확인한다.
다른 작업의 프로세스나 공유 자원을 종료하지 않는다.
자동 충전·재시작·재평가는 하지 않는다.

### 18.2 시간 추정

    total_hours =
      setup + preflight + preparation
      + external_effect_F0
      + sum(main_slot_wall_seconds)/3600
      + extra_development_checks
      + external_effect_final
      + reload_export
      + final_evaluation
      + compass
      + recovery

초기 네 묶음은 경로 검사와 임시 ETA에 사용한다.
대표성 있는 추가 초기 묶음·긴 FULL 행 측정으로 범위를 갱신한다.
가장 짧은 네 묶음만으로 전체 완료시간을 보장하지 않는다.
학습 부모 수 3,000을 자동으로 줄여 ETA를 맞추지 않는다.

### 18.3 평가 계산 예시

기존 B2 출력량 4,116,922token을 단순한 시나리오 입력으로 쓰면:

| 가정한 aggregate 처리량 | 출력 token만의 시간 |
|---|---:|
| 250 token/s | 약 4.57시간 |
| 400 token/s | 약 2.86시간 |
| 600 token/s | 약 1.91시간 |

위 값은 v17 실측 처리량이 아니다.
prefill·모델 로드·긴 tail·검증·저장을 포함하지 않는다.
v17의 출력 길이 분포가 B2와 같다는 보장도 없다.
v15 절반 평가 시간을 단순히 두 배 해서 v17 900 시간을 확정하지 않는다.

14~17시간 목표에 들어가는지는 A1·F0 보존 비용과 안전 동시성의 실제 측정으로 판단한다.
범위를 넘을 것으로 예상되면 시간 breakdown·병목·가능한 최적화와 수정 ETA를 보고한다.
실행 중 자동 양자화·이미지 축소·출력 상한 축소·문항 누락은 없다.

이전의 `예상>15시간이면 시작 금지`와 `15시간이면 종료` 규칙은 폐기한다.
2026-11-15의 3,000문항 전체 학습 결정과 이번 14~17시간 목표를 따른다.
`total_timeout_seconds=null`, `training_elapsed_cutoff=null`을 유지하며 15/17시간 상수를 종료 조건으로 넣지 않는다.
승인된 학습을 ETA만으로 중단하지 않는다. 완료량·새 ETA를 알리고 실행 변경은 사용자 결정에 따른다.
모델 붕괴·무결성 오류·실제 자금 부족 등 사전 합의한 정지 조건은 유지한다.
평가의 기술적 watchdog은 §15의 실측 기반 별도 계약이며, 만료 시 완료 결과를 보존해 재개할 수 있어야 한다.
14~17시간 안에 완료되도록 노력하는 것과 완료를 보장하는 것은 다르다.

목표 시간 배분의 예시는 다음과 같다. 아래는 실측 예측이 아니라 최적화가 맞춰야 할 작업 예산이다.

| 작업 | 목표 배분 |
|---|---:|
| 설치·doctor·짧은 BF16 확인·준비 SFT | 0.5~0.75시간 |
| 본학습 188묶음·마무리 3슬롯 | 8.5~9.75시간 |
| 개발 감시·F0/최종 효과 집합 | 0.5~0.75시간 |
| MMMU 900 연속 처리 | 3.5~4.75시간 |
| 재로딩·내보내기·보완 판정·회수 | 0.75~1.0시간 |
| 합계 | 13.75~17.0시간, 약 14~17시간 목표 |

자료 연결·CPU 회귀·코드 구현은 포드 과금 전에 끝낸다.
이 배분을 지키려고 원래 정밀도·이미지·토큰·문항 수·검증을 줄이지 않는다.
사용자 검토 대기나 외부 장애가 생기면 대기·복구 시간을 별도로 밝혀 전체 경과시간을 보고한다.

### 18.4 비용

실행 직전 실제 잔액·GPU 가격·저장 과금·회수 예약금을 확인해 봉인한다.
사용자는 충전할 예정이라고 밝혔다. 과거 약16달러 잔액을 고정 예산으로 삼아 학습 규모를 줄이지 않는다.
다만 충전 예정과 실제 반영 잔액은 구별하고, 실행 시 충분한 잔액이 확인됐는지 기록한다.
이 작업에서는 잔액·가격을 조회하거나 충전·결제 설정을 변경하지 않는다.
이 문서의 과거 1.39달러/시간을 현재 가격이라고 사용하지 않는다.
추정 비용은 시간×실제 단가+저장 비용이다.
이미 발생한 비용을 회수 뒤 0이라고 하지 않는다.
학습 비용과 평가 미완료 비용을 구분한다.

### 18.5 회수 필수 파일

추론 복원용:

- adapter·Vision adapter·Merger.
- base model/revision·processor 참조.
- checkpoint metadata·export manifest·hash.

동일 학습 재개용 추가:

- runtime.pt의 optimizer/scheduler/RNG.
- cursor와 consumed row/parent IDs.
- runtime drain snapshot.
- request journal·완료 생성 원문·정책 이력.
- baseline·guard·신호 gate 상태.
- 데이터·순서·source·protocol manifest.

v16b는 optimizer runtime.pt가 회수되지 않았다.
그 회수본을 완전한 동일 상태 학습 재개본이라고 부르지 않는다.
v17에서는 “결과만 회수”와 “재개 가능 회수” 모드를 manifest에서 구별한다.

### 18.6 압축·병렬 회수

닫힌 결과를 기존 collect.pack 기반으로 스트리밍 압축한다.
최종 archive를 조각으로 나누고 조각별·전체 hash를 기록한다.
병렬 수는 최초 4, 검증된 경로에서 최대 8로 고정한다.
조각을 합친 뒤 전체 archive hash를 확인한다.
각 조각을 독립 tar로 해제하지 않는다.

로컬 검증:

1. 조각·전체 SHA.
2. 안전한 archive 경로·symlink 정책.
3. manifest와 실제 파일 목록.
4. checkpoint 필수 파일·재개 파일.
5. 완료된 결과의 ID·개수·재채점.
6. LOCAL_VERIFIED receipt.

그 뒤에만 소유 포드·저장 자원을 정리한다.
공유 volume은 삭제하지 않는다.
이번 문서 작성에서는 위 원격 작업을 수행하지 않는다.

## 19. 구현 작업 순서와 담당 경계

### W0. 기준 동결

- 현재 소스·관련 테스트·lock hash 저장.
- 기존 builder/golden 출력 보존.
- 새 v17 파일·변경 허용 파일 목록 확정.
- 분석에 사용한 benchmark/monitor 파생물에 training_allowed=false 유지.

완료 조건: 구현 전후 비교할 기준이 있고 기존 작업을 덮어쓰지 않음.

### W1. 데이터·노출·capability

- v17_data.py 구현.
- 역사적 reservation·해제 근거 감사.
- ChartQA 원표·주석·이미지·가시성·순서 연결기를 선행 구현하고 본학습225개와 적격 미사용 후보를 감사.
- 필요 시 결정적 대체, 역할 수량 고정.
- public/private 분리, 새 source adapter capability 계산.
- data/prep의 v17 분기 연결.

완료 조건: CPU에서 실제 최종 목록과 모든 reason을 재구성하고, 차트 연결 전후 OBS/N 적격 수와 비율을 보고함.

### W2. OBS·NOTE·A1·C

- v17_rows.py 구현.
- ChartQA CHART_VALUE 검증을 generic OBS·실제 수정 N까지 연결하고 미지원 원인을 보존.
- 프로그램 소유 claim ID와 실제 원문 span.
- VERIFIED_REPAIR, lesson, C 독립 receipt.
- 공통 verify_solution와 all-reasons 기록.
- 256 최소 조건 제거, bare answer와 모순 거절.

완료 조건: 실제 과거 원문으로 탈락 이유를 재생할 수 있으며 새 원문을 만들어 성공시키지 않음.

### W3. 손실·마스크

- train_v17.py 구현.
- raw CE와 processed q 구분.
- 고정 분모·빈 채널·마지막 8개·flush.
- OBS 음의 마스크 분리.
- 64위치 F0 KL과 원본 상태 복구.
- component별 신호 확인 경로.

완료 조건: 작은 tensor의 값·gradient가 독립 계산과 일치.

### W4. 캠페인·재개 연결

- protocol.py 새 builder/validator.
- campaign callback의 version 선택.
- encode/backward/joint update dispatch.
- A1 소비·N/C 지연 소비·신호 gate checkpoint.
- 학습16 상한·실제 KV 예약·동기식 제출 대기와 기존 policy/prefill/carry 연결.
- short 추세 guard·전체 길이 분포·gate 시점 외부 감시·정확한 마지막 묶음 분모.
- CLI와 source manifest 포함.

완료 조건: CPU fake-engine 일정과 crash/reload에서 중복·누락·정책 혼합 없음.

### W5. 평가·운영

- v17_eval.py와 proof_eval 등록 연결.
- slots=8, 기존 continuous refill 재사용, 엔진 생성 후 실제 KV 검사, heartbeat.
- 실측 timeout을 plan/신규 preregistration/runner/funding에 같은 영수증으로 연결.
- 같은 START/run의 평가 전용 resume, FINAL_RAW 복구, CPU-only finalize.
- doctor 정확한 argv.
- 실행 장비 watchdog.
- 압축·학습/평가별 필수 재개 파일·샘플 증분 회수 manifest.

완료 조건: 경고가 정상 응답을 중단하지 않고, 용량 실패는 첫 요청 전에 발생하며 완료 답은 재개 후 byte 불변.

### W6. 통합 검사·짧은 실제 확인

- CPU 집중 회귀·Ruff·범위 mypy.
- 독립 검토.
- 승인된 짧은 로컬 NF4 갱신·새 PID reload.
- 별도 승인 뒤 BF16 실제 생성·전송·최대 행 용량 확인.
- A1·64위치 F0 보존의 증가 시간과 8칸 연속 평가 처리량을 별도 실측하고 ETA 영수증 생성.

완료 조건: 기능·교정 신호·운영 검증을 각각 기록.

### W7. 동결·게시·실행

- 최종 source/data/protocol/plan hash 봉인.
- MMDL_test 동기화·게시 검사는 사용자 지시 범위에서 수행.
- clean clone과 고정 commit·lock으로 실행.
- 충전의 실제 반영·회수 예약금 확인, 14~17시간 목표 ETA와 동적 평가 timeout·재개 정책 봉인.
- 15시간 자동 차단이나 학습 문항 자동 축소가 plan/runner/watchdog에 남지 않았는지 검사.
- 초기 신호 gate 통과 뒤 고정 일정 지속.
- 최종 외부 효과·MMMU·회수·정리 결과를 분리 보고.

지금은 W0~W7의 구현을 시작한 상태가 아니다.
이 문서는 그 작업의 계약이다.

## 20. 테스트 설계

### 20.1 새 검사 파일

    fine_tuning/tests/test_error_note_v17_data.py
    fine_tuning/tests/test_error_note_v17_rows.py
    fine_tuning/tests/test_error_note_v17_protocol.py
    fine_tuning/tests/test_error_note_v17_loss.py
    fine_tuning/tests/test_error_note_v17_campaign.py
    fine_tuning/tests/test_error_note_v17_runtime.py
    fine_tuning/tests/test_error_note_v17_eval.py

위 파일은 앞으로 구현할 이름이다.
현재 존재하거나 통과한 것으로 보고하지 않는다.
기존 테스트를 삭제·skip해서 새 계약을 통과시키지 않는다.

### 20.2 데이터 검사

- main/prep/development 수량 3000/512/200.
- historical holdout union과 release receipt.
- ID/native/family/byte/pixel 중복 전파.
- 후보 순서를 뒤집어도 같은 대체 결과.
- 실제 annotation/OBS/N capability가 다른 사례.
- development 원 source_split이 train이어도 학습 차단.
- private annotation을 바꿔도 public schema prompt는 변하지 않음.
- 정답 entity 목록을 노출하는 public catalog 거절.
- NOTE whitelist 밖의 private fact/key/reason이 prompt에 직렬화되지 않음.
- OBS scope·최대 16claim·canonical 768token 초과의 capability 제외.
- 새 효과 자료의 결정적 출처 할당·최소 공급·과거 노출 제외.
- performance-only probe가 main/prep/개발/효과 집합과 family/image까지 분리됨.
- ChartQA225개 전수 capability 원장과 연결 전후 OBS/N 공급 재집계.
- HUMAN 필드 원문·출처 변조 거절.
- 새 텍스트 주석의 오염 검사 receipt 누락 거절.

### 20.3 주장·노트 검사

- claim ID 위조, 다른 generation의 ID, 존재하지 않는 ID.
- OBS o0와 T1 t0가 함께 있는 노트의 교차 참조.
- Unicode byte 경계와 token 경계.
- 중복 key·상충 entity·숫자 NaN/Inf·제어 token.
- 기존 VERIFIED를 오류로 지목한 가짜 repair.
- UNKNOWN→VERIFIED를 repair 성공으로 세지 않음.
- 같은 entity의 실제 wrong→correct 수용.
- 다른 entity로 옮긴 수정 거절.
- N 성공/C 실패, N 성공/lesson 누설의 독립성.
- node/edge 방향·고립 노드·geometry ambiguous join·CLEVR x tie.
- missing fact와 contradicted fact 구별.

### 20.4 풀이 수용 검사

- 짧지만 필요한 근거가 있는 정답 수용.
- 짧은 정답이면서 과정 모순이 있는 사례 거절.
- 긴 모순 풀이 거절.
- 긴 정답 반복만 있는 출력 ANSWER_ONLY.
- 256에서 탈락했던 응답도 뒤의 검사 전부 수행.
- C의 생성 입력과 P0 학습 입력을 별도 hash로 검증.
- 정답 문자열·EOS를 붙여 만든 목표 거절.
- A1 때문에 추가 생성하지 않음.
- PG 제외인 잘린 T1도 검증된 오류가 있으면 NOTE 적격, 원문 전체 SFT는 계속 거절.
- sampled IDs의 decode 일치와 noncanonical 재encode 차이를 구별.

### 20.5 손실 검사

- B=16 고정, 마지막 8개와 빈 slot의 값.
- 수용 N/C가 1개/여러 개일 때 분모 변화 없음.
- 수용 행이 없는 채널의 가중치 재분배 없음.
- raw CE와 processed q가 다른 작은 logits 사례.
- PREP_SELF/C/N은 raw로 변경, HUMAN raw 유지, 준비 마지막 batch 분모와 첫4 anchor 고정.
- presence penalty의 chunk 경계 이력.
- OBS 음의 mask 제거, T1 mask 유지.
- importance weight detach.
- HOLD에서 SFT/KL 포함 모든 gradient 제거.
- 32개 중 단 한 mixed 응답에서도 공동 HOLD, SFT 우회 수용 금지.
- microbatch 분할 전후 loss/gradient 동등성.
- 동일 T1 RL+A1 공유 계산과 분리 계산의 loss/gradient 동등성.
- F0 위치 선택 재현성과 구간 가중치 합 1.
- n≤64에서 dense-position KL과 일치.
- F0에 gradient 없음, Merger/adapter 상태 예외 복구.
- 정답/오답/잘림/PG 제외 T1이 각각 한 번씩 F0 보존에 포함되고 A1으로 중복되지 않음.
- KL-only slot에서 optimizer step 없음.
- 길이 상한 + is_loop → LOOP_CONFIRMED·pg True, 길이 상한 + 비반복 → LENGTH·pg False, 정상 종료 + is_loop → LOOP_CONFIRMED.
- LOOP_CONFIRMED의 PG 범위가 감지 구간(마지막 3,000자) 토큰과 정확히 같고 확률 마스크가 적용되지 않음, 경계 토큰 제외.
- LOOP_CONFIRMED 행의 c_i가 감지 구간 토큰 수로 계산되고, A1·C·PREP_SELF 목표로 쓰이지 않음.
- 저장된 v16b T1 재생 시 LOOP_CONFIRMED 165, LENGTH 86, 정상 종료 오검출 0이 재현됨.

### 20.6 일정·재개 검사

- prepared/queued/prefilled/running/completed의 정책 연결.
- 옛 policy reference가 남으면 덮어쓰기 불가.
- A1/RL 같은 부모의 합법적 동시 학습.
- N/C 지연 소비, 중복 row 거절.
- crash 전후 slot·scheduler·baseline·guard 동일.
- 마지막 세 flush의 빈 상태와 비빈 상태.
- worker 1/2 결과 동일, 처리 순서 독립.
- 신호 gate 판정 뒤 자동 후속 본학습 금지.
- history 없는 통계는 null이며 정상으로 인증하지 않음.
- ref=0/recent=0 무정지, 새 count floor와 마지막 짧은 묶음 표본수 환산.
- 지원 부모 32개 미만의 GPU 진입 거절, 정확한 신호 gate slot/carry 계산.
- short13%→21% WARNING, 13%→30% STOP, WARMUP·재개 후 창/분모 동일성.
- 길이 중앙값이 비슷해도 short와length 꼬리가 함께 늘면 양극화 경고.
- gate 시점 전체main과 외부200의 분포를 혼합하지 않음.
- train KV 예약에 native waiting 포함, 완료당1회 해제, 제출 대기 중 update 금지.
- no-live/no-capacity 상태에서 무한 대기 대신 오류, checkpoint에 미제출 요청 은폐 금지.
- 현재 prompt+상한 예약과 실제 입력 token 일치, 혼합과 단순 선점의 구별.

### 20.7 평가·회수 검사

- KV 용량 경계 바로 아래/위.
- KV 정보 미확인 시 첫 요청 전에 실패.
- 긴 응답이 새 token을 내면 완료 0이어도 진행.
- preemption 증가·새 token 정체 경고.
- 경고가 iterator close/abort/resample을 호출하지 않음.
- registration의 slots와 실제 backend batch size 동일.
- 900개 중 누락·중복 ID면 FULL 거절.
- 불완전 run 둘을 합치지 않음.
- optimizer 없는 archive를 RESUMABLE로 인증하지 않음.
- 조각 hash·전체 hash·안전한 경로·필수 파일 검사.
- 장/단 fake 요청에서 active≤8, 하나 완료 직후9번째 투입, 첫8개 전체를 기다리지 않음.
- 10개 중4개 commit 후 중단·동일 등록 재개: 생성6개, 옛4개 SHA 불변, START1개.
- sample 저장 직후 invocation 갱신 전 크래시: 그 ID 생성0회.
- FINAL_RAW 후 sample 전 크래시: 같은 raw로 commit, 생성0회.
- STARTED-only는 미완료로 남고 기존 부분 출력 보존, 정확 KV/RNG 복원을 주장하지 않음.
-900개 저장 후 summary 전 크래시: CPU finalize만 수행하고 모델 로드0회.
- 완료 run shortcut도 등록·seed·slot·환경 변경을 먼저 거절.
- 새 포드의 kernel/driver/추가패키지/모델 참조 경로 불일치를 조용히 통과시키지 않음.
- measured timeout·funding 예약 일치, 고정23400/15시간 상속 금지.
- samples/·invocation·raw·입력 자산 빠진 분석 archive의 RESUMABLE 인증 거절.
- 개발200과 효과 집합 모두 순변화≤0이면 EVAL_REVIEW_REQUIRED, 효과 미완료를0으로 채우지 않음.
- 병합 export를 평가 입력으로 주면 첫 요청 전에 거절, registration에 unmerged-lora 기록.
- 종료 상태별 분해: 층 기준이 기준 응답(F0·B2)으로 미리 정해지고, v17 결과로 층이 바뀌지 않음. B2 층 90/810 재현.

### 20.8 실행 명령 예시

다음은 구현 후 사용할 명령 형태다.
인터프리터 경로는 현장 manifest로 확정한다.
이 문서 작성 중에는 실행하지 않았다.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
      /home/jang2/mmdl-envs/notebook-nf4-portability-v1/bin/python \
      -m unittest \
      fine_tuning.tests.test_error_note_v17_data \
      fine_tuning.tests.test_error_note_v17_rows \
      fine_tuning.tests.test_error_note_v17_protocol \
      fine_tuning.tests.test_error_note_v17_loss \
      fine_tuning.tests.test_error_note_v17_campaign \
      fine_tuning.tests.test_error_note_v17_runtime \
      fine_tuning.tests.test_error_note_v17_eval -q

연결 회귀는 변경된 영역에 맞춰 기존 아래 모듈을 포함한다.

    fine_tuning.tests.test_error_note_v16b_protocol
    fine_tuning.tests.test_error_note_v16b_logprob
    fine_tuning.tests.test_error_note_v16b_loss
    fine_tuning.tests.test_error_note_v16b_joint
    fine_tuning.tests.test_error_note_v16b_campaign
    fine_tuning.tests.test_error_note_v16b_runtime
    fine_tuning.tests.test_error_note_v16b_resume
    fine_tuning.tests.test_error_note_v16b_plan
    fine_tuning.tests.test_error_note_v16b_eval
    fine_tuning.tests.test_error_note_single_loss
    fine_tuning.tests.test_error_note_single_vllm
    fine_tuning.tests.test_error_note_content_v16
    fine_tuning.tests.test_error_note_data_v16b
    fine_tuning.tests.test_notebook_rl_proof_eval
    tests/test_error_note_v16b_rows.py

마지막 항목은 실제 위치가 다른 검사 파일이다.
현재 test discovery 방식에 맞춰 직접 경로나 해당 suite로 실행한다.
모든 과거 실패를 이번 변경과 무관하다는 이유만으로 PASS에 합치지 않는다.
실행하지 않은 전체 suite는 미실행으로 기록한다.

## 21. 완료 판정

| 단계 | 필요한 증거 | 아직 말할 수 없는 것 |
|---|---|---|
| 설계 완료 | 이 문서·독립 검토·계약 충돌 해소 | 코드 구현 완료 |
| CPU 구현 완료 | 실제 계약·손실·재개·평가 회귀 통과 | 실제 Qwen 학습 |
| 기능 완료 | 실제 생성·갱신·새 PID reload | 교정 능력 개선 |
| 교정 신호 완료 | 실제 wrong→correct N·내용 gradient·갱신 연결 | 새 문제 첫 풀이 개선 |
| 외부 효과 확인 | 동결 외부 집합의 F0 paired 비교 | MMMU 65점 |
| 최종 평가 완료 | 단일 run 900/900·V8·원문·입력·hash | 보지 않은 모든 과제의 일반화 |
| 회수 완료 | LOCAL_VERIFIED와 필수 추론/재개 파일 | 원격 정리 완료 |
| 정리 완료 | 소유 자원 종료·삭제·잔존 확인 | 비용 0 |

학습 데이터 이용률은 별도 지표다.
어떤 직접 신호도 없는 부모가 남을 수 있다.
그 수를 숨기거나 틀린 원문을 SFT로 넣어서 100% 이용률을 만들지 않는다.

## 22. 설계상 남는 위험과 시행 전 확인

1. 결과 조건부 A1/C에는 검증하지 못한 자유 과정 오류가 남을 수 있다.
2. 새 NOTE schema가 실제 Qwen에서 충분한 교정을 생성할지는 미검증이다.
3. 기존 후보 풀만으로는 관측 지원 문항이 저절로 늘어나지 않는다.
4. 과거 reservation의 해제 근거가 확인되지 않으면 지원 문항이 감소한다.
5. OBS whole-response reward는 맞은 일부 claim에도 음의 credit을 줄 수 있다.
6. T1 confidence mask는 이번에도 명시적 편향을 갖는다.
7. 64위치 KL은 전체 응답의 완전한 보존이 아니며 F0 forward 비용은 남는다.
8. prep의 지도·차트 편중은 유지되는 기본안의 한계다.
9. 시각 관측 점수 상승이 P0 첫 풀이로 옮겨갈지는 별도 검증이 필요하다.
10. 안전한 평가 동시성이 기존 짧은 문제 처리량보다 낮을 수 있다.
11. 짧은 기능 시험만으로 14~17시간 완료·예산·585점 달성을 보장할 수 없다.
12. 완전한 v15 비교에는 조상 모델의 학습 노출과 같은 문항 평가가 필요하다.
13. **정상 종료 오답에 대한 직접 학습 신호가 약하다.** v15 오답 170개 중 130개가 정상 종료·답 추출 후 오답으로, 가장 큰 오답 집단이다.
    v17에서 이 집단이 받는 신호는 마스크 뒤 약 16%만 남는 T1 음의 RL(v16b 107,294/684,059토큰)과,
    관측·교정이 가능한 약 5% 문항의 N뿐이다. 마스크 유지는 일반 추론 억제를 피하려는 선택이지만,
    그 대가로 내용 정답률을 올리는 수단이 약하다. 정상 종료 층의 개선은 §5.4 분해로만 판단한다.
14. 반복 감점(§10.3.1)은 생성 낭비와 평가 시간을 줄이는 것이 목적이며, v13 근거로는 점수 향상을 기대할 근거가 약하다.
    지나친 단답화로 이어질 수 있어 §12.4와 함께 감시한다.

이 위험을 줄이는 방법은 실패 기준 완화가 아니라
범위가 명확한 내용 검증, 실제 신호 계측, 원본 대비 첫 풀이 비교다.

## 23. 근거 자료

### 23.1 로컬 코드·문서

- [v16b 회수본 분석](analysis_v16b_recovered_20261116.md).
- [v16b 설계](plan_v16b_self_note_rl_20261115.md).
- [v16b 자료 선정](plan_v16b_data_curation_20261115.md).
- [v16b 구현 인계](handoff_v16b_implementation_20261115.md).
- [학습 경로 README](../fine_tuning/error_note/README.md).
- [현재 상태 원장](../docs/PROJECT_STATUS.md).
- 회수 root: /home/jang2/mmdl-artifacts/self-note-single-20261105/pod-v16b-u1bnytseo804s1/final/.
- CPU 분석 root: /home/jang2/mmdl-artifacts/self-note-single-20261105/v16b/analysis-recovered-20261116/.
- 자료 선정 root: /home/jang2/mmdl-artifacts/self-note-single-20261105/v16b/data-curation-20261115/.

추가 재집계의 실제 입력 경로는 다음과 같다.
아래 B는 설명용 경로 접두어이며 실행 환경변수를 덮어쓰는 명령이 아니다.

    B = /home/jang2/mmdl-artifacts/self-note-single-20261105
    B/v16/data-20261115/frozen/content-manifest.json
    B/v16/data-20261115/frozen-v2/content-manifest.json
    B/v16/data-20261115/frozen-v3/content-manifest.json
    B/v16/data-20261115/frozen-v4/content-manifest.json
    B/v16b/implementation-20261115/data/assembled/dataset-manifest.json
    B/v16b/implementation-20261115/data/assembled/prep-manifest.json
    B/v16b/implementation-20261115/data/assembled/vault/{train,prep,monitor}/
    B/v16b/data-curation-20261115/candidates.jsonl
    B/pod-v15-9n3tu6xbqdk5qd/final/extract/single-v15/campaign/block-{41..56}/rows-16.json
    B/pod-v16b-u1bnytseo804s1/final/extract/runs/v16b/output/campaign/runtime/update-timing-{0..190}.json

candidates의 5,468개는 eligible=true이고 exclusion_reasons가 빈 기존 후보 수다.
v17의 역사적 reservation·family 전파 이후 공급 수라고 소급 표기하지 않는다.
v15의 256문항은 실제 rows의 problem_id 합집합이며 campaign-plan 전체를 사용하지 않았다.
시간 합계는 update-timing의 timing 객체를 합산했다.
other_overhead_seconds는 max(0, wall−generation−update−checkpoint)다.

### 23.2 외부 근거와 적용 한계

- [vLLM 0.11.0 최적화 문서](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html):
  KV 부족 시 선점·재계산이 지연을 늘릴 수 있고 동시 요청·token 예산과 통계가 중요하다는 근거.
  최신 버전의 새 cache API를 0.11.0에 있다고 가정하지 않는다.
- [Visual-RFT](https://arxiv.org/abs/2503.01785):
  검증 가능한 시각 보상으로 학습한 사례.
  해당 연구의 그룹 생성·과제·모델을 우리 단일 응답 MMMU 설정의 성공 증거로 바꾸지 않는다.
- [SCoRe](https://arxiv.org/abs/2409.12917):
  자기 생성 교정을 RL로 학습하는 연구.
  여러 차례 수정 성능과 우리 평가의 첫 풀이 성능을 구별한다.
- [ReST-EM](https://arxiv.org/abs/2312.06585):
  피드백으로 선별한 자기 생성 응답의 반복 학습 근거.
  최종답 정답이 자유 풀이 전체의 정확성 인증이라는 뜻은 아니다.

현재 환경 lock은 Torch 2.8.0+cu128, Transformers 4.57.1, vLLM 0.11.0,
PEFT 0.17.1, Accelerate 1.11.0을 포함한다.
v17을 구현하면서 임의 업그레이드하지 않는다.

## 24. 이 문서 작성에서 수행한 검증

- Git 상태·실제 HEAD·관련 파일 목록 확인.
- v16/v16b 주요 경로의 symbol·본문·호출 계약 분석.
- source 파일의 AST·SHA256·줄 수 확인.
- 회수 분석과 현재 코드의 수치·해석 대조.
- 실제 자료 metadata·OBS/N capability·과거 holdout 겹침 CPU 재집계.
- 고정 vLLM 문서와 설치 버전의 관련 동작 대조.
- 문서의 로컬 링크·기존 소스 경로·기존 회귀 모듈 경로 존재 확인.
- 본학습 3,000·준비 512·미사용 후보 1,756 합계, 188 본학습 묶음, 역사적 KV 계산 재검산.
- 문서 trailing whitespace와 관련 문서 diff 검사를 수행했고 오류가 없었다.
- 독립 설계 검토에서 나온 정책 혼합·확률 공간·분모·claim ID·private 입력·OBS 범위·효과 자료·F0 대상 계약을 반영했다.
- 독립 검토 범위에서 남은 구현 차단 계약은 발견되지 않았다. 코드 실행 인증이라는 의미는 아니다.
- 교수님 MMDL checkout clean과 AGENTS.md/CLAUDE.md byte 동일성을 확인했다.

실행하지 않은 것:

- v17 소스 구현·production 테스트·lint/typecheck.
- GPU·모델 생성·학습·reload·성능 평가.
- RunPod 조회·접속·게시·실행·삭제.
- 데이터 최종 재선별·새 manifest 봉인.
- 과거 holdout 해제 권한의 완전한 확인.

따라서 이 문서의 상태는 IMPLEMENTATION_PLAN이며 IMPLEMENTED/PASS가 아니다.
