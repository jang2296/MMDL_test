# v17 구현 인계 — Claude Code 검토용

작성일: 2026-11-17. 원 계획은 `plan_v17_final_implementation_20261116.md`다.
계획 원문 SHA256은 `0899453deb1b5dd3f351977589feb6a55d4d6a5cdfde0a754a73ee607c78cd27`이며, 구현 프로토콜이 참조하므로 원문은 보존했다.
이 문서는 구현 결과와 실제 실행 전 남은 검증을 구분한다.

## 1. 인계 범위

- 사용자가 요청한 v17 코드 변경, 자료 동결, CPU 검사, 짧은 로컬 NF4 시험을 수행했다.
- RunPod 조회·접속·변경, 모델 다운로드, Git commit/push, MMDL 동기화는 수행하지 않았다.
- 상위 작업공간의 feature branch `feat/mmmu-baseline`에서 기존 미커밋 변경을 보존했다.
- 기존 `src/mmdl`·V8·교수님 MMDL·과거 해시 고정 파일은 작업 시작 시점과 대조한다. Git HEAD와의 전체 diff에는 이번 작업 이전 변경도 섞여 있다.
- 과제 점수는 B2/V8 558/900 그대로다. v17의 MMMU 성적과 65점 달성 여부는 아직 없다.

## 2. 구현한 학습 동작

| 구분 | 구현 |
|---|---|
| 시작 | 지정 Qwen3-VL-4B-Instruct 원본 F0, 새 optimizer/scheduler |
| 자료 | 본학습 3,000, 별도 준비 후보 512, 학습 제외 감시 200 |
| 생성 | 정답 없는 T1 1회, 적격 OBS 최대 1회, NOTE 최대 1회, 누설 없는 교훈 재풀이 최대 1회 |
| T1 RL | 실제 생성의 처리 후 전체 어휘 확률, 사전 baseline, 기존 지연 일정 유지 |
| 반복 | `LOOP_CONFIRMED` 감지 구간 음의 PG. 일반 음의 확률 마스크 미적용, 행 길이 계수 적용 |
| 단순 잘림 | 반복 근거 없는 LENGTH는 PG 제외. 생성량과 제외 사유를 보고 |
| OBS | 공개 문법만 모델에 제공하고 원 주석은 private verifier에서 사용 |
| A1 | 이미 생성한 정답 T1을 검사 후 보조 SFT에 재사용. 추가 생성 없음 |
| C | 256토큰 하한 제거. 정답·모순·반복·누설·답만 있는 응답 검사는 유지 |
| N | 실제 자기 주장 ID의 틀린 값→검증된 수정값 연결. 수정 내용 토큰에만 SFT |
| 독립 수용 | 재풀이 실패와 N 수용을 분리. 교훈 누설과 검증된 수정 내용도 분리 |
| 손실 | SFT는 raw CE, RL은 처리 후 분포. 본학습 채널 분모 16 고정, 준비는 기존 별도 계약 |
| F0 보존 | T1 응답 전반 최대 64위치, 층화 선택·구간 가중치. 원본 전체 prefix forward 비용을 별도 측정 |
| 계산 공유 | 같은 T1의 RL+A1 현재 모델 forward 공유, 행·receipt·mask는 각각 유지 |
| 무결성 | 혼합 정책 응답이 포함된 공동 갱신은 HOLD. KL만으로 학습 성공 처리하지 않음 |
| 시각 경로 | Vision LoRA·Merger·LLM LoRA 실제 손실 경로 유지. 전체 기저 가중치 학습으로 표시하지 않음 |

틀린 원문을 자동 SFT로 넘기지 않는다. 모순이 발견되지 않았다는 이유만으로 자유 풀이 전체의 정확성을 인증하지 않는다.
답·관측 보상, A1·C·N 수용, 실제 N gradient와 checkpoint 반영을 따로 기록한다.

## 3. 코드 위치

| 파일 | 책임 |
|---|---|
| `fine_tuning/error_note/v17_data.py` | 과거 예약 자료 격리·대체, capability, private 주석 연결, 결정적 순서, 효과·성능 전용 자료, portable 묶음 |
| `fine_tuning/error_note/v17_rows.py` | 공개 OBS/NOTE 계약, 실제 토큰·claim ledger, 검증·누설·A1/C/N admission |
| `fine_tuning/error_note/train_v17.py` | RL/OBS/A1/C/N/KL, 고정 분모, 공유 forward, N gradient, BF16 메모리·비용 측정 |
| `fine_tuning/error_note/v17_preflight.py` | 실물 사전 점검 영수증, 실행 identity 바인딩, 최종 비병합 서빙 검사 |
| `fine_tuning/error_note/v17_eval.py` | 8칸 연속 평가, 실제 KV, 측정 시간 한도, heartbeat, 동일 등록 재개·CPU finalize·회수 |
| `protocol.py`, `plan.py` | 새 v17 식별자·손실 설정·준비/본학습 실행 명령. v16b 계약 별도 유지 |
| `prep_v16b.py`, `campaign_v16b.py` | 명시적 v17 분기, 기존 생성·학습 호출 사슬 재사용 |
| `single.py`, `single_campaign.py`, `train_v4.py`, `contract.py` | 상위 진입점과 encoder/backward·seed 연결 |
| `v16b_runtime.py`, `single_vllm.py` | 기존 두 정책·η=1 일정, KV 예약 기반 제출 제어, heartbeat |
| `guard_v16b.py` | 점진적 단답화·길이 양극화·실제 N 신호 gate, v17 자금 계산 |
| `lora_preflight.py`, `llm_lora.py` | F0의 0 조정층과 학습 후 효과 검사를 구분, 비병합 적용 |
| `fine_tuning/notebook_rl/proof_eval.py` | v17 사전등록 v12, 비병합·교정된 평가 wrapper 연결 |
| `fine_tuning/notebook_rl/pod/collect.py` | 평가 재개 파일을 포함한 압축 archive 및 검증 |
| `fine_tuning/notebook_rl/pod/verify.py` | v17 manifest를 실제 v17 validator로 검사하는 명시적 버전 분기 |

다섯 번째 새 helper인 `v17_preflight.py`는 실제 장비 검사와 영수증을 한곳에 두기 위한 분리다. 별도 학습 프레임워크나 모델 다운로드 경로를 만들지 않았다.

## 4. 데이터와 현재 한계

- 과거 네 reservation manifest의 합집합 530개를 확인했다. 현재 본학습과 겹친 32개를 격리하고 기존 적격 후보에서 결정적으로 대체했다.
- 본학습 3,000 / 준비 512 / 감시 200 / 외부 비교 128 / 성능 측정 전용 16은 서로 다른 역할이다. 감시·외부 비교·성능 측정 자료를 학습에 쓰지 않는다.
- 학습 순서는 저장된 `order.json`의 실제 ID 순서를 그대로 읽는다. 출처·기존 F0 난도·capability·긴 응답 위험을 함께 섞는다.
- 현재 엄격한 범위에서 본학습 OBS 대상은 262개, 교정 지원 대상은 387개다. 그래프 OBS 231, 장면 OBS 31, 도형은 교정 지원 125이지만 표시 사실 전체 범위 인증 부족으로 OBS 0이다.
- ChartQA 연결기는 구현했다. 하지만 기존 적격 후보의 엄격한 위치·가시성·원표 연결 검사에서 실제 OBS 적격은 0이다. 차트 225개가 본학습에 있다는 사실을 차트 관측 교정 225개라고 보고하면 안 된다.
- 외부 비교 128개는 과거 노출 이력을 완전히 확인하지 못해 `EXPOSURE_HISTORY_INCOMPLETE`, `effect_eligible=false`다. 새 일반화 효과의 확정 근거로 쓰지 않는다.
- 동결 자료에는 `performance-requests.json`과 해당 이미지도 포함한다. 모델에 전달하는 관측 정답 문장을 프로그램으로 만들어 학습하지 않는다.
- 전송 helper는 symlink·hardlink를 거절한다. v17 freeze는 독립 inode의 byte 동일 이미지 사본과 hash를 저장하고 실제 bundle 경로 검사까지 수행한다.

차트 지원 0과 효과 자료의 노출 불완전성은 필터를 완화해 해결하지 않았다. Claude는 이 제한을 실제 실행 판단과 결과 보고에 포함해야 한다.

최종 데이터 root는 다음 하나다. `assembled`·`assembled-r2/r3/r4`는 중간 검토 기록이며 실행 입력으로 사용하지 않는다.

```text
/home/jang2/mmdl-artifacts/self-note-single-20261105/v17/implementation-20261117/data/assembled-r5
```

| 파일 | SHA256 |
|---|---|
| `payload-receipt.json` | `3edb302dd388e2ddc98cd6258ff47bd2ce9a69fea901aab5cda3dc1170a9aa2c` |
| `dataset-manifest.json` | `691bb93cc757c7187f4e724a3976335fc62e048dc8fd9af5ec7d40402a68373a` |
| `bundle-spec.json` | `bad5a15d85177ca395b493cd4dabc9c13f6b4a30bb6bf3ecdfeb405aae980f78` |
| `performance-requests.json` | `ff9fe34c49d3300c38e3f7482fb138bf8cc3123c2ecd71039f682043969ec57c` |
| `order.json` | `bb6d91ef5d1cc83b8ae16a2d8377cfdbbbe0d24f5fb1568d8f7437f75789c1a6` |

실제 전송 목록 7,747개에는 `bundle-spec.json` 자체와 성능 측정 이미지가 포함된다. `validation.json`은 로컬 감사 기록이며 전송 필수 파일은 아니다.
최종 자료의 역할·사실·순서와 capability는 다시 확인했다. 과거 원자료 불변 검사는 r3에서 완료한 증거를 명시적으로 이어받았으며, r5에서 같은 검사를 새로 했다고 표시하지 않았다.

## 5. 관문·감시·재개

- 최초 지원 부모 32개의 NOTE/RESOLVE가 닫힌 뒤 gate를 평가한다. 실제 N 내용 기울기와 저장된 갱신에 연결된 교정 부모를 센다.
- FAIL/INCONCLUSIVE는 다음 묶음 생성 전에 정지 상태를 저장한다. 검증된 교정이 없는 채 3,000개를 끝까지 소비하지 않는다.
- gate 측정 중 장애가 나도 재개 시 gate/대기 중 감시를 먼저 처리한다. PASS 저장 전 또는 FAIL 저장 후의 crash를 각각 검사했다.
- 초반 24묶음과 최근 12묶음의 짧은 응답 비율, 길이 histogram·분위수·8192 도달·최종답 누락을 감시한다. 마지막 8문항의 실제 분모를 사용한다.
- 묶음별 단계·출처별 생성/수용/거절 수를 저장한다. 음의 T1 후보·잔류 토큰과 반복 감점 토큰을 별도 기록한다.
- 감시 200, 외부 비교, 최종 MMMU 비교 모두 F0의 잘림/반복 집단과 정상 종료 집단을 나눈다. 끝맺기 변화와 내용 정답률 변화를 구분하기 위한 관측 비교다.
- 현재 외부 효과 자료는 확인된 미노출 집합이 아니므로 최종 평가 진입에 검토가 필요할 수 있다. `evaluation-admission.json`을 확인한다. 필요한 사용자 결정은 실제 결과와 결합된 `evaluation_decision` 영수증으로 남긴다.

## 6. 시간·평가·회수

- 14~17시간은 목표다. 이 숫자로 본학습을 자동 차단하거나 3,000문항을 줄이지 않는다.
- 학습은 기존 두 정책·η=1 일정과 flush를 유지한다. 긴 응답 대기를 없앴다고 주장하지 않는다.
- 학습 제출 상한은 16이며, 실제 KV의 90% 안에서 입력+출력 cap을 예약한다. 실행/대기 요청 모두 예약에 포함하고 여유가 생겨야 제출한다.
- 평가는 동시 8칸이다. 한 요청이 끝나면 나머지 긴 요청을 기다리지 않고 다음 ID를 넣는다. 정오 점수로 순서·재시도를 선택하지 않는다.
- 실제 KV가 8개 최악 문맥을 감당하는지 첫 평가 요청 전에 검사한다. 이미지 크기·토큰·정밀도를 자동으로 줄이지 않는다.
- heartbeat는 누적 생성 토큰, 활성/대기 수, KV, 선점, 긴 요청 진행을 기록한다. 180초 완료 없음은 경고이고 곧바로 답을 버리는 기준이 아니다.
- 평가 시간 한도는 성능 전용 16개 이하의 실제 wall time/생성 토큰과 보수적 시나리오로 봉인한다. 짧은 표본만 측정했으면 LOW_COVERAGE를 유지한다.
- F0 KL과 A1 비용은 실제 원본 forward·현재 모델 forward·출력 projection·backward로 나눈다. A1 수용 0이면 측정 성공을 꾸미지 않는다.
- 평가 재개는 같은 등록·모델·seed·동시성의 미완료 ID만 처리한다. 완료 오답·잘림도 보존한다.
- `FINAL_RAW`를 yield 전에 원자 저장한다. 저장 raw가 있으면 GPU 생성 없이 sample을 복원하고, 900개가 있으면 CPU finalize만 한다.
- 복구 동등성은 원문·token·채점·identity 등 안정된 필드에 대해 검사한다. 저장하지 못한 원래 timing을 지어내지 않고 복구 timing을 별도 표시한다.
- 시간 한도는 최초 논리 실행부터 누적한다. 재개가 시간을 초기화하지 않는다. 승인된 연장은 append-only 영수증으로 연결한다.
- 최종 평가는 `model_ref.json`과 합치지 않은 조정층으로만 서빙한다. 병합 모델은 v17 첫 요청 전에 거절한다.
- 평가 재개 archive에는 완료 sample, FINAL_RAW, 입력 자산, registration/ledger, 모델·설정 identity가 포함된다. 회수 파일 manifest만 만들고 안전한 외부 사본이 생겼다고 주장하지 않는다.

## 7. 실행 명령 연결에서 수정한 결함

1. 동결 순서가 실제 학습 순서에 전달되지 않던 부분.
2. gate checkpoint 뒤 crash에서 다음 학습으로 넘어가던 부분.
3. EOS 전용·빈 출력의 claim audit 예외. 실제 action 0개면 PG도 제외한다.
4. 존재하지 않는 간선의 올바른 `false` 주장 오판, 도형 선분 끝점 순서 정규화.
5. 차트의 숫자 값과 unit 공개 문법 불일치.
6. 전송할 freeze 이미지의 hardlink 및 성능 전용 요청 파일 누락.
7. 자료를 풀기 전에 그 안의 성능 요청 파일을 pin하던 명령 순서.
8. 평가 `--base-path`, 캠페인 `--code-commit`, 실제 preflight 실행 옵션 누락.
9. 성능 probe의 8칸/in-process 엔진 환경 변수 누락.
10. export 파일명 `model-ref.json`과 실제 `model_ref.json` 불일치.
11. calibration의 두 파일 hash와 본학습의 전체 소스 hash 불일치. 전체 `single.source_manifest().actual_files` 범위로 통일했다.

단순 함수 단위 mock 통과에 그치지 않도록 실제 parser·고정 영수증·파일명을 사용하는 CPU 연결 검사를 추가했다.

## 8. 로컬 실제 시험

- 기존 Qwen4B NF4 cache만 사용했다. 새 모델 다운로드 없음.
- 전체 180초 상한 안에서 실제 T1 한 건, v17 encoder/backward, optimizer 1회, 624 tensor checkpoint 저장을 수행했다.
- L/M/V 기울기는 각각 약 `9.11e-5 / 5.90e-5 / 2.44e-5`로 유한·비영이었다. 정책 hash가 바뀌었다.
- 시험 응답은 4토큰이었다. 이 작동 시험으로 긴 풀이, 교정 N/C, 본학습 효과를 인증하지 않는다.
- checkpoint 저장 뒤 private 관찰 스크립트에서 scalar Adam step hash 오류가 났다. 원래 실패를 보존하고 관찰기만 수정했다.
- 같은 시간 상한 안에서 새 프로세스 재로딩을 시도했으나 timeout 124였다. `UPDATE_SAVED_RELOAD_UNVERIFIED`로 기록한다. 추가 생성·추가 학습·시간 연장 없음.
- 소유한 시험 프로세스는 종료됐다. 실제 BF16/vLLM, 전체 캠페인 재로딩, N 학습·MMMU 효과는 미검증이다.
- 시험한 소스 스냅샷은 해당 폴더의 `source-manifest.json`에 고정했다. 이후 최종 연결 수정까지 포함한 전체 소스를 GPU 검증했다고 확대하지 않는다.

영수증: `/home/jang2/mmdl-artifacts/v17-implementation-20261117/local-smoke/`.
`functional-receipt.json`, `inspection-receipt.json`, `generation.json`, `loss-receipt.json`, `observer-failure.json`, `checkpoint/`를 함께 본다.

## 9. Claude Code의 후속 순서

1. 이 인계 문서·원 계획·이번 변경만 비교한 source receipt를 읽고 검토한다. 본문에 있는 제한을 PASS로 바꾸지 않는다.
2. 최종 데이터 root와 payload hash, 실제 순서, 성능 전용 자료, 과거 reservation 제외를 확인한다. 기존 v16b freeze를 덮어쓰지 않는다.
3. v17 guard의 F0 bootstrap 영수증을 확인한다. 필요한 기존 tail/IS 증거가 없으면 INCOMPLETE로 남긴다. 옛 버전 영수증을 새 PASS로 재표기하지 않는다.
4. 사용자 지시에 따른 게시 시 기존 `sync_test_repo` 검사·선택 파일 diff·clean MMDL_test commit 절차를 사용한다. 이번 작업에서는 게시하지 않았다.
5. 실제 RunPod 실행은 Claude의 후속 사용자 실행 지시와 실제 잔액·가격·소유 자원 범위를 확인한 뒤 한다. 로컬 cache·모델·코드를 포드에 overlay하지 않는다.
6. 기존 `data_bundle` 도구로 private 자료를 한 번 stage/검증한 뒤 승인된 `start.sh`를 실행한다. clean 고정 commit의 `v17-calibration` 체인은 doctor→이미 stage된 자료 검증→준비 자료→BF16 사전 점검→성능 probe를 수행한다. `start.sh` 자체가 자료를 먼저 확인하므로 체인 안에서 중복 stage하지 않는다.
7. 실제 기능·비용·평가 calibration 영수증의 파일 SHA를 사용하여 `v17` 본학습 체인을 새로 봉인한다. 가짜 값이나 로컬 NF4 영수증으로 BF16 요건을 대신하지 않는다.
8. calibration job과 main job은 별도 job 실행 디렉터리와 체인 로그를 사용하되, 공유 출력 부모와 동결 자료는 동일하게 유지한다. 기존 chain 로그를 다른 spec으로 덮어쓰지 않는다.
9. 신호 gate/감시가 정지하면 원인과 실제 영수증을 먼저 확인한다. 900평가로 자동 넘어가지 않는다.
10. 최종 admission이 REVIEW이면 실제 결과에 대한 결정을 받은 뒤 그 영수증을 `require-complete/export/preflight`에 연결한다. 새 seed나 다른 답으로 통과시키지 않는다.
11. 최종 비병합 검사와 재로딩 후 같은 등록으로 900 평가→보완 판정→회수·hash→소유 자원 정리 순서를 따른다.

계획 생성 진입점은 다음과 같다. 아래는 명령 형식이며 실행한 명령이 아니다.

```bash
python -m fine_tuning.error_note.plan plan \
  --stage v17-calibration --params /PRIVATE/calibration-params.json --output /PRIVATE/calibration-steps.json
python -m fine_tuning.error_note.plan plan \
  --stage v17 --params /PRIVATE/main-params.json --output /PRIVATE/main-steps.json
```

params의 정확한 키는 `plan.py`의 `PARAMS_VERSION_V17`, `V17_PATHS`, `V17_HASHED`, `V17_FUNDING_KEYS`, `_check_v17`이 정본이다.
`guard_manifest_path`, `performance_requests_path`, `performance_model_ref_path`, 실제 calibration/functional receipt와 SHA가 필요하다.
사전 점검 결과를 보기 전에 본학습 calibration SHA를 만들어 넣지 않는다.

## 10. 원격 감시와 비용의 책임 범위

- 기존 `pod/start.sh → job.financial → fine_tuning.budget` watchdog을 재사용한다. 노트북·SSH가 끊겨도 초기 실제 잔액과 승인한 금융/GPU 종료 경계를 적용한다.
- 기존 job은 유한한 `max_main_seconds`를 요구한다. 이 값은 승인된 실제 금융/GPU 경계에서 정하며 14~17시간 목표를 기계적으로 넣는 값이 아니다.
- `funding_decision_v17`은 측정 평가 시간과 후처리·회수·storage·예약금을 명시적으로 사용한다. v16b의 5.65시간 가정을 v17에 재사용하지 않는다.
- 초기 snapshot은 이후 실제 계정 잔액 조회가 아니다. 실제 잔액·포드 과금 상태의 10분 간격 점검은 Claude 운영 감시에서 수행한다.
- 기존 watchdog의 작업 프로세스 종료와 포드 과금 정지는 다르다. 회수 검증 후 소유 포드 종료/삭제와 잔존 확인을 별도로 수행한다.
- 자동 충전·자동 재시작·추가 GPU·유료 영속 저장소를 만들지 않는다.

## 11. 완료 수준

이번 인계의 대상은 구현과 짧은 검증이다. 실제 BF16 실행 가능성·교정 수율·전체 학습 속도·MMMU 585/900은 아직 검증하지 않았다.
차트의 실물 검증 범위, 효과 자료 노출 이력, 초기 신호 gate, 실제 장비 calibration이 실행 판단에 영향을 줄 수 있다.
검토 후 실제 실행에서 실패하면 원인과 기존 원문을 보존하고, 기준을 낮추거나 완료 답을 다시 생성하지 않는다.

## 12. 최종 검사와 재현 증거

| 검사 | 실제 결과 |
|---|---|
| v17 + 연결된 기존 경로 22개 모듈 | **332 tests OK**, unittest 114.46초, 프로세스 wall 약121초 |
| v17 데이터·깨끗한 전송 경로 | **14 tests OK**, 약154초. 실제 pack→원본 경로 제거→stage→validator→포드 validator subprocess 포함 |
| Ruff | 변경 소스·v17 검사 파일 통과 |
| 범위 mypy | 캠페인·준비·계약·진입점·새 손실/행/사전 점검/평가/자료·포드 검증/회수 11개 파일 통과 |
| 보호 파일 | 작업 시작 시 보관한 111개 보호 파일 hash 불변. V8·기존 src/mmdl·MMDL 포함 |
| 지침 파일 | `AGENTS.md`와 `CLAUDE.md` byte 동일 |
| 독립 검토 | 발견된 실행 연결·재개·자료 전송 결함 수정 후 남은 P1 없음. 실제 GPU/클라우드 인증은 아님 |

최종 통합 명령과 로그:

```text
/home/jang2/mmdl-artifacts/v17-implementation-20261117/final-validation/combined-tests.json
/home/jang2/mmdl-artifacts/v17-implementation-20261117/final-validation/combined-tests.log
```

변경 전 스냅샷은 같은 구현 artifact root의 `before/`, 최종 변경 목록·해시는 `final-validation/source-review.json`에 둔다.
실행하지 않은 저장소 전체 테스트를 통과로 합산하지 않았다. 기존 protocol 일부의 범위 타입 오류는 앞선 검사에서 확인되어 전체 프로젝트 mypy PASS라고 보고하지 않는다.
최종 통합 검사는 `CUDA_VISIBLE_DEVICES=''`로 실행했다. 로그의 CUDA 플랫폼 탐지나 합성 A40/32칸 출력은 기존 CPU fixture이며 실제 포드 실행이 아니다.

작업 완료 후 Codex는 대기한다. Claude Code가 검토하고 후속 실행을 맡는다.
