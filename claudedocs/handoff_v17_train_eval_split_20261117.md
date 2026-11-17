# v17 학습·평가 분리 구현 인계 — 2026-11-17

상태: A·B 코드 구현과 로컬 검증 완료. Claude Code 검토·게시·RunPod 실행 인계 대기.

계획 정본: `claudedocs/plan_v17_train_eval_split_20261117.md`.
이번 범위는 A·B 전체 구현과 짧은 로컬 검증이다. 새 학습·MMMU 평가·GPU·RunPod·게시·모델 다운로드는 하지 않았다.

## 1. 변경 구조

| 계획 항목 | 구현 위치 | 동작 |
|---|---|---|
| A1 | `fine_tuning/error_note/plan.py` | `v17-train`·`v17-eval`과 실측 후 평가 continuation 생성 |
| A2 | `v17_preflight.py`, `campaign_v16b.py` | `bind_training`·`bind_evaluation`, 명시적인 split 실행 분기 |
| A3 | `v17_artifacts.py` | 체크포인트·내보낸 모델·학습 코드·재로딩·기능·비용·평가 결정 봉인 |
| B1·B2 | `v17_eval_engine.py` | 최종 비병합 모델의 graph/eager 검증과 자동 선택 |
| B3·B4 | `v17_eval.py`, `notebook_rl/proof_eval.py` | 실제 KV 동시 수, 지연 요청 제출 순서, 원장·재개 결합 |
| B5 | `v17_eval_engine.py`, `plan.py` | 실제 길이 분포 처리량으로 ETA·별도 종료 한도 계산 |
| B6·§9 | `plan.py`, `fine_tuning/budget.py`, `notebook_rl/pod/job.py` | 예상 금액 부족은 기록, 실제 잔액 snapshot 기준 예약금 보호 |
| §5 | `notebook_rl/pod/start.sh`, `verify.py`, `chain.py` | clone root·commit 대조, 평가 continuation 별도 로그 |
| 재개·회수 | `notebook_rl/pod/collect.py` | 새 모드·동시 수·제출 원장·검증 증거를 회수 묶음에 보존 |

`src/mmdl/`의 기존 파일, V8 파서, P0, BF16, 32K 상한, sampling, 학습 자료·학습 알고리즘은 변경 대상이 아니다.
기존 `v17-calibration`·`v17`·v16b는 별도 레거시 분기로 유지한다.

## 2. 두 고정 checkout

1. 학습 clone `/root/MMDL_test`에서 고정 commit으로 학습 준비부터 새 프로세스 재로딩·내보내기·봉인까지 마친다.
2. 평가 clone `/root/MMDL_test_eval`의 commit은 같아도 되고 달라도 된다. 두 clone을 모두 보존한다.
3. 평가가 시작될 때 원래 학습 코드·계획·체크포인트·내보낸 모델·관련 영수증을 다시 확인한다.
4. 프롬프트·V8·입력 변환·seed·B2 설정·MMMU scorer·사전등록 계약도 별도로 대조한다.
5. 학습 중인 clone이나 완료된 봉인 파일을 `git pull`, 덮어쓰기, 해시 재작성으로 바꾸지 않는다.

영수증의 JSON 내용 digest와 파일 SHA256은 구분한다.
`training_artifact_sha256`·`training_artifacts_file_sha256`은 직렬화된 영수증 **파일 SHA256**이다.

## 3. 학습 계획

공통 params:

- `version`: `error-note-plan-params-v17-split-v1`
- `commit`, `python`, `repo_root`, `model_path`, `out_root`, `run_dir`, `hourly_usd`, `gpu_type`
- `funding`, `timeouts`

학습 전용 params:

- `data_root`, `bundle_manifest_path`, `bundle_archive_path`
- `guard_falsealarm_path`, `guard_manifest_path`, `functional_receipt_path`
- `bundle_manifest_sha256`, `guard_falsealarm_sha256`, `guard_manifest_sha256`
- `training_estimate_seconds`, `training_estimate_basis`
- `timeouts`: `cpu`, `calibrate`, `reload`, `export`

`training_estimate_seconds=26208`은 v16b RL 7.28시간의 기록용 예시다.
준비·기능 확인·새 SFT/F0 비용을 포함한 확정 전체 시간으로 사용하지 않는다.
출력·기능 영수증 경로는 새 run 안에 둔다. 과거 영수증 해시를 새 코드의 영수증인 것처럼 재사용하지 않는다.

```bash
python -m fine_tuning.error_note.plan plan --stage v17-train \
  --params /private/v17-train-params.json --output /private/v17-train-spec.json
```

생성된 체인은 준비 생성 → BF16 기능 확인 → 본학습 → 평가 결정 → 완료 확인 → 새 프로세스 재로딩
→ 비병합 export → `training-artifact-seal` 순서다. 평가 속도 측정은 없다.
기능 확인에서 실제 A1 비용을 충분히 관측하지 못하면 그 상태와 누락 길이 구간을 보존한다.

## 4. 평가 계획

학습 봉인 이후 실제 파일 SHA를 구해 평가 params를 만든다.

- 경로: `training_artifacts_path`, `export_root`, `hardware_path`, `eval_data_root`,
  `b2_predictions_path`, `b2_lengths_path`, `performance_requests_path`, `cv_script_path`
- 파일 SHA: `training_artifacts_sha256`, `hardware_sha256`, `b2_predictions_sha256`,
  `b2_lengths_sha256`, `performance_requests_sha256`, `cv_script_sha256`
- `eval_engine_mode`: 기본 권장 `cuda_graph` (검증 실패 시 eager 자동 검증)
- `long_tail_allowance_seconds`, `long_tail_basis`
- `timeouts`: `cpu`, `calibrate`, `validate`, `compass`

```bash
python -m fine_tuning.error_note.plan plan --stage v17-eval \
  --params /private/v17-eval-params.json --output /private/v17-eval-spec.json
```

평가 체인은 먼저 최종 모델 검증을 수행하고, 결과로 continuation의 실제 모드·칸 수·watchdog을 봉인한다.
처리량이나 미래 영수증 SHA를 임의로 만들어 사전등록하지 않는다.
continuation은 별도 `eval_continuation_log.jsonl`로 실행되며 이후 MMMU 900·V8·CompassVerifier 단계로 연결된다.

### GPU에서 확인하는 것

- 기존 PF5와 실제 여러 decode 단계의 조정층 증폭 효과.
- 외부 성능 16문항의 eager 대비 raw logprob 차이와 기존 동기화 임계값.
- graph 실행 시 실제 replay 관측. 단순히 `enforce_eager=False`라는 설정만으로 인증하지 않는다.
- 실제 sampler의 raw logprob 모드. 제한된 후보 집합의 확률 1을 동등성 근거로 쓰지 않는다.
- 실제 engine 설정·KV 용량, `floor(0.9 * KV / 40960)` 동시 수.
- 장문 32K 문맥·선점 0과 별도 길이 분포 측정.

각 엔진 검증은 별도 프로세스에서 수행해 graph 실패 뒤 eager가 이전 GPU 메모리를 물려받지 않게 한다.
graph 실패 사유·모든 시도 영수증을 남긴다. eager도 실패하면 MMMU를 시작하지 않는다.
900문항 시작 이후에는 코드·모드·동시 수·등록을 바꾸어 이어 붙이지 않는다.

### 진단의 기술적 시간 한도

`timeouts.validate`는 `validation_timeout_seconds(timeouts.calibrate, mode)` 이상이어야 한다.
probe cap 2700초·자동 선택의 기술적 상한은 18180초다. 이는 graph 실패·eager 재검증·기준 비교·정리까지
가능하게 하는 **최대 한도**이며 예상 소요시간이나 새 5시간 대기 목표가 아니다.
전체 14~17시간 완료와 graph 속도 향상은 실측 전이다.

## 5. 이번에 준비한 길이 입력

비공개 로컬 파일:

`/home/jang2/mmdl-artifacts/v17-split-implementation-20261117/b2-output-lengths.json`

- 파일 SHA256: `9863b44d236c2fe51a4605c3cb486a728308ca8f493722db823b56a04b67b5ac`
- 900개, 생성 토큰 합 4,116,922.
- 질문·정답·이미지·응답 원문 없음. 성능 진단 전용이며 학습에 사용하지 않는다.
- 기존 private 자료 전달에 이 작은 파일을 포함하고 평가 params에 실제 목적 경로를 지정한다. 공개 저장소에는 게시하지 않는다.

16개의 고정 분위수 길이로 진단 출력을 생성한다. 이 진단 응답은 답 선택이나 평가 점수에 사용하지 않는다.
실제 길이 진단 처리량을 ETA와 보수식에 사용하며, 장문 스트레스 처리량은 메모리 검사·진단으로만 남긴다.
900문항의 실제 길이와 일치한다는 보장은 없다.

## 6. 자금·충전·정리

`funding` 필드:
`balance_usd`, `balance_observed_at`, `first_bill_at`, `storage_usd_per_hour`,
`recovery_reserve_usd`(1.5), `recovery_seconds`.

새 pod job은 `financial_mode=balance_reserve_only`와 실제 실행 clone의 `repo_root`를 기록한다.
예상 비용 부족으로 시작을 거절하거나 `max_main_seconds`를 줄이지 않는다.
기술적 단계 제한과 평가 watchdog은 유지한다.

잔액은 실제 관측 snapshot이다. watchdog 자체는 RunPod 잔액 API를 조회하지 않는다.
Claude는 실제 잔액·청구·진행을 확인하고, 충전이 반영되면 명시적인 연장 영수증을 추가해야 한다.

```bash
python -m fine_tuning.notebook_rl.pod.job /path/to/run/job.json \
  --extend-financial --balance-snapshot-utc ACTUAL_UTC --balance-usd ACTUAL_BALANCE \
  --topup-reference RECEIPT_REFERENCE --authorization USER_AUTHORIZATION --confirmed-by claude
```

이 명령은 실제 소유 Pod ID 환경과 고정 job을 대조한다. 기존 job·학습 코드를 수정하지 않는다.
예약금 경계에서는 작업을 중단하고 durable `POD_STOP_REQUIRED` 영수증을 남긴다.
**작업 프로세스 중단은 과금되는 Pod의 stop이 아니다. 실제 Pod stop·회수·삭제는 Claude 운영 단계다.**
결과 보존·회수 확인 전에 Pod를 삭제하지 않는다.

## 7. 중단·재개·회수

- 완료된 `FINAL_RAW`는 같은 등록으로 재사용한다. 완료 문항은 다시 생성하지 않는다.
- 실제 요청 순서는 `sha256('v17-eval-order:' + id)`이며 seed 계산은 그대로다.
- 제출 의도·수락은 해시 연결 원장으로 남고, 실제 inference invocation ID와 연결된다.
- 회수 묶음에는 동시 수·모드·제출 원장·학습 봉인·엔진 검증 시도·PF5 영수증과 증거 경로 매핑이 들어간다.
- 원래 봉인 JSON의 절대 경로·해시를 재작성하지 않는다. 새 Pod에 복원한다면 검증된 증거 바이트를 매핑의 원래
  소유 경로에 복원한 뒤 재개한다. 학습 clone·평가 clone·원본 모델·최종 export/checkpoint는 별도로 유지해야 한다.
- 평가 회수 묶음은 `EVALUATION_ONLY`다. 모델 가중치·optimizer를 포함한 학습 재개 인증이 아니다.

## 8. 로컬 검증 결과

최종 분리 검사 **469개 통과, 기존 조건부 건너뜀 1개**. 중복 실행한 개별 검사는 합계에 다시 더하지 않았다.
증거 디렉터리:
`/home/jang2/mmdl-artifacts/v17-split-implementation-20261117/`.

| 검사 | 결과 | 증거 |
|---|---|---|
| v17·v16b 핵심 회귀, 새 split 검사 포함 | 290개 통과, 테스트 48.57초 / 프로세스 54.61초 | `final-core-stable.json`, `.log` |
| v17 자료 검증·포장·복원·Pod 연결 | 14개 통과, 5.33초 | `linux-data-tests-receipt.json`, `.log` |
| 기존 Pod·chain·회수·budget·vLLM 연결 | 165개 통과, 기존 조건부 건너뜀 1개, 프로세스 98.65초 | `final-operations-tests.json`, `.log` |
| Ruff | 변경 Python 소스·새 테스트 통과 | `static-checks.json`, `final-fixture-lint.log` |
| 범위 mypy | 변경 소스 12개 통과 (`--follow-imports=skip --ignore-missing-imports`) | `static-checks.json` |
| Bash 문법·Python AST·추가 줄 공백 | 통과 | `static-checks.json`, `source-review.json`, `whitespace.json` |
| 보호 대상 | 시작 시 기록한 68개 파일 해시 불변, AGENTS/CLAUDE byte 동일, MMDL clean | `before-manifest.json`, `source-review.json`, `final-validation.json` |

정확한 실행 argv는 각 검사 JSON의 `command`에 보관했다. Python은
`/home/jang2/mmdl-envs/notebook-nf4-portability-v1/bin/python`, CPU 검사는
`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:.`로 실행했다.
건너뜀은 기존 `test_planned_runs_are_in_the_real_preregistration`의 `local preregistration not present` 조건이다.
검사를 새로 disable하거나 기준을 낮추지 않았다.

자료 검사는 `/mnt/c`에서 반복적인 소스 해시 읽기로 180초 제한을 넘겼다.
동일한 현재 소스·설정 685개(14,363,151바이트)를 Linux 경로에 복사하고 파일별 SHA를 확인한 뒤,
수정·mock·해시 캐시 없이 같은 14개 검사를 실행해 통과했다. 실행 후 복사본 불변도 확인했다.
증거는 `linux-cpu-source-manifest.json`이며 SHA는
`ec568bfbc7e2ffc58a0ea15fcc6e3ecd88960df159436f5ec21104867c4ab944`다.

통합 중 발견한 테스트 환경 격리 문제는 실제 환경 스키마와 실제 `environment_identity`를 쓰도록 고쳤다.
기존의 모델·커밋·소스·환경 불일치 거절 검사는 모두 유지했다.
전체 `test_notebook_rl_proof_eval`까지 넓힌 최초 34모듈 실행은 420초 제한으로 미완료이며 통과로 세지 않는다.
초기 타입 오류·테스트 오류·시간 제한 로그도 보존했다. 전체 저장소 검사나 전체 의존성 타입 검사 통과를 주장하지 않는다.

실제 포드 기록 16개를 사용한 calibrate CPU 하네스도 92.21초 통과했다.
원래 Qwen 프로세서·요청 seed·행 생성·인코딩 코드를 사용했고 GPU 함수만 대체했다.
17개 행(RL 16, A1 1)이 구성됐으며, 중·장문 A1 비용 미측정은 `CALIBRATION_LOW_COVERAGE`로 보존됐다.
이 결과는 실제 GPU 학습·속도·모델 개선의 인증이 아니다.

계획 17항목과 구현·검사의 대응은 `requirements.json`, 현재 변경 파일 해시는 `source-review.json`,
작업 시작 복사본과의 diff는 `implementation.patch`다. 독립 소스 검토에서 확인된 P1/P2 잔여 결함은 없었으며,
검토자가 GPU나 테스트를 실행한 것으로 기록하지 않았다.

## 9. Claude의 후속 작업

1. 이 문서·계획 정본·최종 검증 영수증·변경 diff를 검토한다.
2. 지정한 짧은 검사를 확인하고 MMDL_test 게시 절차를 수행한다. 교수님 MMDL은 변경하지 않는다.
3. 실제 커밋·환경·자료·잔액으로 두 stage의 params/job을 생성한다.
4. 승인된 RunPod에서 설치와 기존 압축 자료 업로드를 겹치고 학습 체인을 실행한다.
5. 같은 Pod의 별도 평가 clone에서 최종 모델 검증과 900 평가를 수행한다.
6. V8·보완 점수·미완료 여부·소요시간을 구분해 보고하고 회수·해시 확인·자원 정리를 수행한다.

Codex의 이번 작업에는 Git commit/push, RunPod 접근, GPU 실행, 7B 다운로드가 포함되지 않는다.
