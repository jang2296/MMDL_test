# Assignment 1 baseline — 작업 상태

## 2026-10-04 15:52 UTC — full10K 준비 종료·실제 GPU F0 평가 진입

f3 고정 commit의 첫10K processor 검증은 통과했으나 near-image 검토 후보25개를
학습에서 제외했다. 정상 processor/권리/오염검사를 통과한 TRACE25개로 보충했으며
부모9975개 원장·제외 목록·manifest는 byte 불변이다. near hash 일치를 실제 오염으로
확정하지 않는다. complete_pilot1924.334초/15:41:31UTC, smoke root 분리318.832초/
15:46:50UTC, MMMU60 cohort26.039초·외부60 cohort15.160초/15:47:31UTC에 exit0으로 닫혔다.

F0 MMMU 평가 PID3135는15:47:43UTC에 시작했고15:52:30UTC4/60응답을 저장했다.
실제 A40 GPU83%·10,039MiB를 확인했다. 이는 새4,096출력 진단 프로토콜의 추론이며
학습 또는 과거32K558/900 성적의 갱신이 아니다. 다음은 외부 F0→네 조건의 학습 smoke→
실측2배 예산 gate→같은 full10K 네 조건이다. 전체 G6/G7/transfer 완료는 아직 아니다.
원래 잔액10.77/reserve1USD/금융 종료 한도·모델/샘플링/데이터 조건을 유지한다.

내부 `analysis_exports/g6_preparation_receipts_20261004/`에 닫힌 metadata·실행 f3 코드667파일을
보존했다. 원격/로컬 hash·승인 연결·Git blob/실행권한을 검사했으며 최종 이미지/학습노출/
체크포인트/평가 전체 회수·검증은 미완료다. 이 부분 회수만으로 Pod를 삭제하지 않는다.
MMDL_test의 미래 CPU 최적화는 실행 f3에 적용하지 않았고 교수님 MMDL/V8은 불변이다.

## 2026-10-04 15:00 UTC — 10K 후보 변환 완료·다음 실행용 CPU 승인 중복 제거

같은 소유A40의 `approve_pilot` PID2456은 실행 중이며 GPU0MiB/0%다.
TRACE 변환이14:41:46UTC에 닫혔고14:42:10UTC에10,000개 `proposed_records.jsonl`이
생성됐다. 변환 원장은13,015개, 이미지16,765개이며 후보10K 승인과 다른 지표다.
실제 processor preflight/최종 approval 완료는 아직 확인되지 않았으며 학습 전이다.

다음 실행용 `audit`의 eager fallback set 생성을 제거하고 processor receipt를 최초 읽은
bytes로 파싱·hash하여 모든 행과 manifest에 같은 digest를 연결한다. 승인 완료 직전에
원본 receipt를 다시 hash하여 변경되면 출력 디렉터리 생성 전에 거부한다. 원본 index에서
캐시를 검증·구성하는 단계, producer/source/processor/revision·입력/권리/오염/coverage
검증은 유지한다. 관련40 CPU tests(29.762초)·Ruff·bounded mypy1·diff 및 독립 검토 PASS.
자체 CPU fixture의 결과이며 실제 processor 실행이나 cloud 가속률 측정이 아니다.
아래14:31UTC 절에서 미수정으로 기록한 두 hotspot은 이 별도 미래 수정에서 처리했다.
실행 중인 f3 checkout/원장/승인 hash는 불변이며 새 코드로 실행하려면 새 정상 승인이 필요하다.
교수님 MMDL·V8 558/900 불변, full10K/G7/transfer 미완료 및 원래 비용/시간 한도를 유지한다.

## 2026-10-04 14:31 UTC — GPU 대기의 실제 단계·미래 CPU 중복 계산 제거

같은 소유A40의 CUDA/BF16와 지정 모델은 정상이나 GPU0MiB/0%이며 아직 학습하지 않는다.
준비 controller의 benchmark 취득100파일/16,740행, 배제 index368.733초, TRACE24.344초,
smoke360개 CPU 승인127.288초가 exit0으로 닫혔다.14:24:06UTC 시작한 `approve_pilot`
PID2456은14:31UTC TRACE Parquet 읽기와 PNG 쓰기를 수행하며 이미지5,569개를 생성했다.
이 수는 승인된 학습 행 수/정답률이 아니다. 전체10K 승인·보충·smoke root 분리·두 평가
cohort를 준비한 뒤에만 GPU controller를 시작하며, 거기서도 F0 평가가 학습 smoke보다 먼저다.
GPU 장애나 학습 중 낮은 utilization이 아니라 설치/검사/데이터 준비를 직렬 실행한 지연이다.

다음 실행의 `sources.benchmark_index`는 동일 raw SHA를 이미 성공적으로 decode한 경우에만
pixel/dHash 계산을 재사용한다. 모든 이미지 schema/count/raw SHA·파일 hash·행 coverage는 유지한다.
실제 기존 index는19,041이미지 출현/12,168고유 raw SHA/6,873중복이다. 미래 중복 호출 감소의
근거이지 전체 소요시간 단축 실측이 아니다. 자체 두 Parquet의 같은 byte/다른 encoding·동일 픽셀,
corrupt/missing 이미지, per-file coverage와 hash 반례를 확인했다. 실행 중인 f3 코드와 원장은 불변이다.
추가 CPU 승인에서 cached set의 eager fallback 및 행마다 전체 processor receipt 재해싱도 확인했지만
이번 수정에 포함하지 않았다. 향후에는 검증을 생략하지 않고 동일 hash 결합을 유지해야 한다.
교수님 MMDL 및 V8 558/900은 불변이며 full10K/G7/transfer는 여전히 미완료다.

## 2026-10-04 14:07 UTC — 고정 수정본 setup 통과·G6 준비 실제 재개

`f3f37ab624cfa6d35cc8a7cc85ba95cbac553ce1`를 MMDL_test에 게시하고 같은 소유A40 Pod의
clean checkout에서 실행했다. 새 `/workspace/nvw-runtime-retry1`을 사용하며 이전 venv만
재사용했다. exact lock/pip check·전체180 CPU tests(591.522초)·실제 CUDA/BF16 연산을 통과했다.
모델 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`의13개 파일 hash를 확인하고
setup은749.724초/exit0으로 닫혔다. 이전 실패 로그/원장은 보존한다.

G6 controller는14:05:46UTC 시작했으며 prototype 생성50.201초·검증14.865초/exit0을
기록했다.14:07UTC에는 새 pilot pool 생성 중이다. 아직 학습/새 평가/전체10K 완료는 아니다.
원래 과금 시작·잔액10.77·reserve1USD·상한0.55USD/h·금융 종료 한도는 재설정하지 않았다.
10분마다 실제 Pod API/프로세스/GPU/오류를 확인한다. GPU0%인 현재 준비 단계를 학습으로
표기하지 않는다. 증거 `analysis_exports/g6_cloud_deployment_20261004.json` 및 원격
`/workspace/g6-control/launch-retry1-f3f37ab.log`, `runs/g6_launch/journal.jsonl`.
교수님 MMDL·V8 558/900·기존 응답/체크포인트는 불변이며 실행 중인 f3f37ab 코드를 갱신하지 않는다.

## 2026-10-04 13:35 UTC — setup 회귀 시간 가정 실패·학습 진입 차단

소유A40 `qg7krba8ebl10s`의 고정 pip 설치·pip check는 완료했으나 전체177 CPU tests는
176통과/1오류(552.173초)로 종료했다. watchdog의 실제 종료 테스트가0.2초 안에 중첩 Python의
PID 출력이 나온다고 가정해, 안전 종료 JSON을 PID로 읽었다. watchdog 자체는 BUDGET_STOP과
소유 child cleanup을 수행했다. GPU doctor/모델 취득/G6 학습에는 진입하지 않았다.
원격 프로세스211/213/219/434 종료와 GPU0MiB를 확인했다. 실패 로그·pip freeze를
`analysis_exports/g6_setup_failure_20261004/`에 보존했다. 실패를 PASS로 바꾸지 않는다.

테스트는 의도적인0.35초 cold-start 지연과5초 유한 timeout으로 수정하고,
별도0.05초 빠른 timeout 검사·실제 worker 종료/회수 검사는 유지한다.
로컬 관련21 CPU tests(10.298초)와 Ruff를 통과했다. 실제 금융/시간 watchdog 값은 변경하지 않는다.
새 GitHub exact SHA로 닫힌 checkout을 갱신하고 새 runtime에서 검사를 다시 실행한다.
설치된 venv 재사용은 정상 lock 설치/pip check/전체검사/doctor/모델 hash 검사를 생략하지 않는다.
원래 과금 시작13:05:01.730UTC/잔액10.77/reserve1/상한0.55h/최종 한도는 유지한다.
10분마다 소유Pod API/단계/오류/예산을 확인하며 기존 Controller의600초 HEALTH_POLL도 기록한다.
과거 속도로는 보수적2배 G6 비용 gate를 통과하기 어려우며 실제 새 smoke/F0 측정으로 판단한다.
교수님 MMDL·V8 558/900·기존 학습 결과는 불변이고, 전체10K 완료는 아직 미검증이다.

## 2026-10-04 13:10 UTC — MMDL_test 게시·새 A40 설치 재개

게시 commit `9882aacde28cf27e4c35772ea9032acf7e3aedb9`와 원격 feature HEAD를 확인했다.
MMDL_test666파일/31,883,243bytes의 source/hash/mode 동등성, checkout22 CPU tests/4.202초,
Ruff/Bash/diff·exact SHA setup/G6 dry-run을 검증했다. 교수님용 MMDL은 clean·`197838c…` 그대로다.

새 소유Pod `qg7krba8ebl10s`/EU-SE-1/A40 1개를 생성·read-back·직접 SSH 연결했다.
GPU $0.49/h,30GB container/160GB persistent `/workspace`, network volume 없음.
실제 A40 46068MiB/driver580.159.04/Python3.12/cgroup RAM49,999,998,976bytes,
CPU quota7.65코어를 확인했다. CPU OMP/MKL threads8을 적용한다. shared MFS의 df를 Pod quota로 오인하지 않는다.
설치부터 포함한 all-in 상한$0.55/h·초기잔액$10.77·회수 reserve$1을 고정한다.
금융 한도2026-10-05 06:50:50.820909UTC가 요청한18시간 종료보다 먼저 적용된다.

첫 배포는 CPU RAM64GB 조건의 인스턴스 없음으로 실패했으며 목록0을 확인했다.
새 재고 확인 뒤 CPU RAM32GB 조건으로 생성했다. 첫 launch는 SSH 비로그인 셸의 실제
RunPod ENV 누락으로 파일 생성 전에 안전 중단됐다. PID1의 실제 Pod ID/OMP ENV와
RunPod가 만든 `/etc/rp_environment`를 확인해 source한 별도 launch로 재개했다.
코드 overlay/런타임 코드 수정은 없으며 GitHub exact SHA만 clone했다.
로그 `/workspace/g6-control/launch-env-v2.log`. 실제 학습/전체10K/새 점수는 아직 미완료다.
회수·검증한 뒤 이 소유Pod만 삭제한다. 증거 `analysis_exports/g6_cloud_deployment_20261004.json`.

## 2026-10-04 — 작업 재개·cold help 수정·fresh G6 실행 경로

사용자 '계속 진행해'로 재개했다. RunPod 실제 목록0개, A40 48GB SECURE는
EU-SE-1 LOW·GPU $0.49/h로 확인했다. 이 값은 디스크 포함 가격이나 잔액이 아니다.
MCP에 잔액 조회가 없어 현재 금액을 질문했고 사용자가10.77USD를 확인했다.
회수 reserve1USD를 남기며 설치/준비도 전체 금액·시간 한도에 포함한다.

평가 CLI help를 heavy runtime import보다 먼저 처리해30초 preflight의 cold-import 경로를
제거했다. 모델/생성/V8 채점은 불변이다. runtime import를 금지한5개 새 subprocess를 포함해
평가/새 launch 관련16 CPU tests 및 targeted Ruff를 통과했다.
새 `fine_tuning.runpod_g6`는 clean MMDL_test full SHA와 새 setup receipt를 요구하고
고정 자체 생성 pool → benchmark exclusion → 정상 현재-code 승인 → smoke root 배제 →
full10K controller를 기존 구현으로 연결한다. 준비도 시간/금액 한도에 포함하며 내부 GPU
controller는 outer watchdog보다2분 먼저 닫는다. 새 Cloud/GPU 실행·새 점수는 아직 없다.
일반 READY 원장의0-delta completion과 기존 quarantine completion 관련5 CPU tests를
통과했다(219.946초; 실제 processor/GPU 검증 아님). 독립 검토에서 발견한 setup 한도 미적용과
leader 종료 후 잔존 worker 경로를 공통 watchdog/child cleanup으로 수정했다.
중첩 SIGTERM에서 외부45초/내부15초 grace를 분리하고 실제 소유 CPU worker의 종료·회수를
검사했다. 관련18 tests/3.616초·전체 실험177 CPU tests/406.510초·게시 안전3 tests,
targeted Ruff·budget/launch bounded mypy2·Bash 구문 검사를 통과했다.
최종 독립 검토에서 해당 변경의 차단 문제는 발견되지 않았다. 새 GPU 성공은 아직 미검증이다.
완료 판정은 실제 full exposure/reload/paired evaluation에 따르며 작은1600/Graph 결과로 대체하지 않는다.

## 2026-10-04 — MMDL_test 게시·실행 경로 전환

사용자가 A40 Pod를 삭제했고 계정 Pod 목록0개를 확인했다. `jang2296/MMDL_test`는
공개/빈 저장소·push 권한을 확인했다. 제출용 MMDL은 변경하지 않는다.
실험 코드·문서·설정·필수 자체 생성 CPU fixture와 과거 v18 source를 별도 checkout에
동기화하며 비밀키·가중치·HF/benchmark 원문·대형 결과·강의 원본은 게시하지 않는다.
새 bootstrap은 고정 commit MMDL_test clone에서 실행하고 수동 overlay를 요구하지 않는다.
설치/CPU 검사/GPU doctor/모델 취득까지만 하며 학습은 별도 승인 입력을 요구한다.

삭제 전 원격 common10K 준비 READY를 관측했지만 controller는11:14UTC
`python -m fine_tuning.evaluate --help`의30초 timeout으로 BLOCKED, GPU 단계 미실행이었다.
최종 원격 데이터/승인 전체는 회수되지 않았으며 이전 TRACE completion metadata만 로컬에 있다.
기존1600 실험과 V8 558/900은 보존한다. 새 bootstrap GPU 성공과 전체10K 완료는 미검증이다.
MMDL_test checkout 실험166 CPU tests(309.096초)/게시 안전3 tests/Ruff/mypy2/Bash PASS.
게시 selector·bootstrap 독립 검토를 완료했다. vendored 코드/기존 Markdown의 역사적 공백은
원문 hash 보존을 위해 유지했으며 이번 변경 파일의 diff 공백 검사는 통과했다.

## 최종 로컬 보존·자원 정리 — 2026-10-03 15:18 KST

기능적 진단30과목102문항·Knowledge15분야60호출의 실제 결과는 보존·검증됐다.
추가 Gemma는06:07:38UTC GPU한도에 종료했다.900행 중 실제응답749개(558자동/110검토/
81오류),151개는시간한도(중단1/미시작150)이며 새 정확도는null이다. 자동558은 정답수가 아니다.
부모729개 무변조 재사용·새20실응답,749입력/native token-ID와668판정·81오류를
main/독립 검증했다. 부모89payload+continuation19payload의 원문·환경·코드를 로컬에 보존했다.
소유Pod `re5qljwwyq7yui`는06:18UTC 삭제204→조회404→목록부재, network volume0이다.
`owned_gemma_segments_cleanup_final_v1.json` 참조. 추가 추론·유료자원·학습·게시를 시작하지 않는다.
기존558/900·V8·공개 MMDL/GitHub는 불변이다. 미실행과 의미 검증 실패를 성공으로 표기하지 않는다.
최종 보고서: `claudedocs/Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md`.
결과: `analysis_exports/bottleneck_20261003/{export_all30_102_v2,knowledge_scores_v1,gemma_segments_final_export_v1}`.

## 실제 Gemma 재개 — 2026-10-03 05:58 UTC

최초 source-segment run은729개 실제 응답(550RESOLVED/102REVIEW/77SYSTEM_ERROR)에서
원래 기한 종료·소유 서버 정리를 완료했다. 부모 원장/입력/manifest는 변경하지 않는다.
별도 samePod continuation에 원래48파일 불변·최소8파일 v3 SHA·실제Pod16회귀·새 서버
model/version/native sampler를 확인했다. 로컬16검사/Ruff/mypy2와 독립 Sol delta검토도 PASS.
729개 실제 응답을 재사용하고171개만 이어 실행한다. 미반환 오류를 완료 응답으로 상속하지 않는다.
GPU한도06:07:38UTC/최종회수·삭제06:22:38UTC, 기존558/900·MMDL/GitHub 불변이다.
새 유효 전수 점수는 아직 없다. 부모89파일은 닫힌 archive로 포장했으며 로컬회수·검증 중이다.
`private_continuation_code_v3.receipt.json` 참조. 최신 상태는 이 절을 아래 역사 기록보다 우선한다.

## Gemma 재검증 승인·구현 — 2026-10-03

사용자가 추가2시간과 새3090 1개를 승인했다. 새 종료06:22:38UTC/15:22KST,
GPU 종료05:52:38UTC/14:52KST. 과거 실패와 baseline은 보존한다.
별도 source-segment-citations 프로토콜로 원문 인용 생성 오류를 막되 의미·철회·충돌 가드는 유지했다.
로컬900개 손실 없는 입력/gold독립 검사 PASS. 실제CPUtokenizer900·GPU29gate를 통과하고
04:00UTC경 전수감사를 시작했다. CPU최대122159+출력1024=123183/문맥131072이며 원문절단0.
내부 구현·검증 후 새Pod에서만 재검증하며 MMDL/GitHub/학습/기존응답은 변경하지 않는다.

03:41:44UTC 새Pod `re5qljwwyq7yui`에서48파일의 비공개 bundle을 hash검증 후 설치 시작했다.
RTX3090/24GB/driver580.159.03·고정 base digest/Python3.12.3/workspace mount 확인.
70 Gemma CPU tests(49base+14source+7controller 분할실행)·24 exporter/controller tests·4 recovery tests/Ruff/mypy/Bash PASS.
새 프로토콜의 실제GPU900 완료/새점수는 아직 없다. 이전실패162/8/5/1은 원문 그대로 보존한다.

04:22UTC 닫힌81건은61RESOLVED/11REVIEW/9SYSTEM_ERROR다. 전체입력7,820,217token은
이전5,748,953 대비약36% 증가했다. 근거를 host가 정확한 원문 위치에서 복원하므로 비원문 인용
생성 경로는 제거했지만, 잘못된/빈/중복 행ID·답과 근거 불일치·JSON출력잘림은 계속 차단된다.
전수 완료가 기한을 초과할 가능성이 있어05:52:38UTC GPU종료와06:22:38UTC 회수종료를 유지한다.
새로운 전체 정확도는 미확정이며 기존62.00%는 불변이다. 닫힌 실제gate13파일을 로컬 회수해
CPU900 입력·hash를 재현했다. GPU29PASS는 기계적 계약 통과이지 의미 정확도 인증은 아니다.
근거: `gemma_segments_gates_main_verification_v1.json`,
`gemma_segments_input_cost_comparison_v1.json`, `cloud_gemma_segments_gates_v1/recovery_receipt.json`.

04:30UTC 추가검증:70 Gemma검사를 한 번의 discover에서 모두 PASS(32.315초), Ruff/mypy5/Bash PASS.
`segments_local_cpu_validation_v3.json`은 기존 분할검사 기록을 덮어쓰지 않는 후속 증거다.
닫힌첫81 입력·실제token-ID·응답·72terminal판정·9실패를 main에서 재현하고 별도 Sol도 검증했다.
같은81의 이전58RESOLVED/15REVIEW/8ERROR→새61/11/9이며, 자동판정률 변화이지 정확도 개선은 아니다.
새9실패는 judge1024출력잘림3/빈행2/중복ID2/답표기 원문불일치2다. host위치 결합 오류는
이 표본에서 발견되지 않았지만 유효ID가 의미적 정답·충돌 판정의 진실성을 보장하지는 않는다.
`gemma_segments_prefix81_main_verification_v1.json`,
`gemma_segments_prefix81_failure_independent_v1.json`, `gemma_segments_gates_independent_verification_v1.json` 참조.

05:04UTC 원래 immutable run은398/900(328RESOLVED/38REVIEW/32ERROR) 진행 중이다.
사용자 승인 최종06:22:38UTC 안에서만, parent가 닫히고 소유GPU cleanup이 확인된 경우
같은Pod의 hash검증된 완료 원장을 별도 continuation에 상속하는 절차를 준비한다(DESIGNED).
기존GPUcutoff05:52:38·원장/manifest는 수정하지 않으며, 별도 재개의 GPU최대06:07:38UTC와
최종15분 회수 여유를 둔다. 모델/원문/추출프로토콜/생성값은 불변, 완료문항 재생성·새Pod 없음.
새 실행의 현재 코드·기존CPU900/29gate·freshserver staticidentity·원문/입력/record key를 확인한
뒤에만 상속한다. 미확정/미시도 budget행은 완료 응답으로 상속하지 않는다. 아직 재개 실행은 없다.

05:19UTC 닫힌첫420 실제기록을 로컬 회수·전체 payload/실제 token-ID·판정과45실패를 재현했다.
330RESOLVED/45REVIEW/45SYSTEM_ERROR, 실패는 답-원문불일치21/출력계약12/빈행8/중복ID4다.
입력2만 미만371건은325/21/25,5만 이상32건은1/15/16이다. 장문에서 미확정·가드거부가
집중되는 관측이지 인과효과나 정확도 인증은 아니다. `gemma_segments_prefix420_main_verification_v1.json` 참조.
별도 continuation13 CPU회귀·전체83 Gemma검사/Ruff/mypy2·독립 Sol검토 PASS.
최소추가8파일+원래48파일만으로 격리 import를 검증했고 원래48 SHA는 불변이다.
전송 archive의 hash·8파일 inventory는 `private_continuation_code_v2.receipt.json`에 있으며,
실제 parent 종료/소유cleanup 및 freshserver 검증 후에만 재개한다. 실행 성공을 미리 주장하지 않는다.

## 비공개 진단 실행 종료 — 2026-10-03 03:03:58 UTC /12:03 KST

- Qwen102/30과목·Knowledge60/15분야의 실제 실행·회수·검증은 완료했다. 원래558/900·62.00%
  baseline과 공개 MMDL은 변경하지 않았다. K1/K2 각30/30은 더 단순한 다른 문항의 최종답이다.
- Gemma900시도는02:39:37 UTC 종료,900추출+2동치 응답을 보존했다.607RESOLVED/117REVIEW/
  176SYSTEM_ERROR,293검토 큐이며 유효900판정은 INCOMPLETE·새 정확도null이다.
  실제 실패162비원문인용/8답-근거불일치/5Gemma출력1024잘림/1잘못된JSON을 CPU 재현했다.
  실패를 Qwen오답으로 세거나 원문 가드를 완화하지 않는다. Music_5는 유효 회수 후보,
  Public_Health_28은 모순 때문에 REVIEW이며 둘 모두 점수에 자동 반영하지 않았다.
- 최종42payload+manifest archive와 실패 이력23payload를 로컬 회수·byte/hash검증했다.
  기존 Qwen/Knowledge/환경을 포함한18개 로컬 증거와 독립 검토를 삭제 gate에 연결했다.
  `cloud_gemma_final_v8/local_receipt.json`, `gemma_final_recovery_independent_v1.json`,
  `gemma_final_candidates_independent_v1.json`, `final_handoff_independent_v1.json` 참조.
- 정확한 소유Pod h447a4u6c0vq3k를03:03:45 UTC 삭제204→조회404→목록0 확인했다.
  30GB container/200GBworkspace 폐기, 별도NetworkVolume 생성/삭제 없음·조회0.
  Gemma전체가중치는 로컬 회수하지 않았다. 코드·원문·원장·환경·실패 로그는 보존했다.
  `owned_pod_cleanup_final_v1.json`은 CLEANUP_VERIFIED이며 과학적 목표 성공 인증은 아니다.
- 실제 청구 snapshot10bucket GPU1.9209609823301435/disk0.2795138885267079/합USD2.2004748708568513.
  반영 지연 가능·최종청구서 아님. 관측장치최대22,768MiB·truepeak아님.
- 108 private CPU검사/Ruff/4파일mypy 및 exporter9검사 기존 결과를 유지했다. 현재 변경은 내부
  문서·종료 증거뿐이며 FT/새유료실행/공개MMDL/GitHub변경 없음. 잔여160/추가seed/LEGACY/
  구조완전성·유일모듈책임은 미측정이다. 두 내부 claudedocs 보고서에 종료 상태와 한계를 반영했다.

## 비공개 진단 체크포인트 — 2026-10-03 02:14:52 UTC /11:14 KST

계획서9질문의 현재 답을 내부 보고서§8에 보강했다. 독립 검토의11hash·35모드 사례·4전이표를
`analysis_exports/bottleneck_20261003/report_completion_main_verification_v1.json`으로 재현했다.
O1-only회복6개·O1오답38(PARSED25/NO_PARSE13)·최종조건NO_PARSE136/647은 실제 원장 집계다.
P전체 정확성 인증과3seed 반복은 미측정이며, FL/FV/FM는 조건부 후속 후보이지 학습 성과가 아니다.

실제 Gemma642/900(451RESOLVED/70REVIEW/121SYSTEM_ERROR), controller7819/birth7298428 동일.
종료 파일은 `/workspace/artifacts/private-finish-v8/summary.json`이고 초기조회의
`controller_summary.json` 파일명은 잘못됐다. 해당 부재를 종료·미종료의 근거로 사용하지 않는다.
원문/런타임/가드 불변, full900회수·검증·소유Pod삭제는 아직 미완료다.
49+59 private CPU tests/Ruff 및 로컬src기준4파일 mypy 통과.
새 청구 v2는 반환9bucket 합USD1.9485288984142244이며 최종 청구가 아니다.
GPUcutoff03:52:38 UTC/최종회수04:22:38 UTC; FT/MMDL/GitHub 변경 및62% 대체점수 없음.

## 2026-10-03 01:50 UTC /10:50 KST — 닫힌 감사128 원인 재현·후속 집계 수정

- 첫128완료 entry를 닫힌사본으로 회수해 ID/payload/input/entry/실제 usage와 모든 판정·실패를
  재현했다.91RESOLVED/22REVIEW/15SYSTEM_ERROR이며 아직 전체900의 결과가 아니다.
- 독립 gold-blind 원인 검토와 main의18인용·source offset/hash 확인:13문항은 근거 문구
  변형(단순 표기뿐 아니라 다른 option label 포함),2문항은Gemma1024출력에서 JSON잘림이다.
  실제 실패를 정상 근거로 치환하거나 Qwen 오답으로 합치지 않는다. 현재 가드는 올바르게 거부한다.
  근거: `gemma_prefix128_failure_analysis_v1.json`, `gemma_prefix128_failure_main_verification_v1.json`.
- post-run exporter의 semantic mapping 요청/HTTP raw/선택 결과 재현과 단계별 비용을 보강했다.
  큰 원문 source해시의 큐당 재계산을 frozen digest 재사용으로 바꿨고 최종2변조 검사는 유지한다.
  9 CPU tests/Ruff/mypy2 PASS, 실제128의 구·신 summary/queue byte동일 확인.
  `gemma_prefix128_export_v2_hash_reuse/local_receipt.json`, exporter SHA `edcbfd7c…21184f`.
- 추론 runtime/preflight/controller hash·900 source·558/900·MMDL은 그대로다.
  최신 실제조회312/900,229RESOLVED/34REVIEW/49SYSTEM_ERROR; controller7819/birth7298428 살아 있다.
  GPU cutoff03:52:38 UTC/최종회수04:22:38 UTC 유지. 전체 완료·의미점수·Pod삭제는 미완료다.
- private README의900 gate/고정 코드 단위/미실행262잔여·반복·LEGACY를 현재 사실에 맞췄다.
  미래 재현에는 새 승인 deadline과 실제 identity를 써야 하며 현재 내부 파일을 공개하지 않는다.

## 2026-10-03 01:30 UTC /10:30 KST — 실제 smoke 회수·전수 감사의 제한 확인

- 통과한29 GPU smoke의 닫힌6파일을 로컬 회수·원장 재현했다. 모든 실제 payload/usage,
  code/CPU/model/source identity와 Math_12 전체50454 입력 token hash가 일치한다.
  근거: `cloud_gemma_passed_smoke_v8/local_receipt.json`.
- 독립 실제 응답 검토도 동일29 machine-outcome과 계약·hash를 확인했다.
  `gemma_passed_smoke_v8_independent_review.json`에 의미 witness의 오분류 및 reason의
  계산/환산 오류를 보존했다. 이는 AI 보조 검토이며900 의미 판정 인증이 아니다.
- 29/29는 고정 출력·안전 실패 계약의 통과다. 의미 witness 자체는 일부 부정확하고
  단위 환산/계산을 금지한 시험의 reason에도 계산이 나타난다. 해당 최종 선택은 null이지만
  이를 의미 정확도 인증으로 확장하지 않는다. live 코드/정책을 수정하지 않는다.
- 실제900감사 중88개 확인:59RESOLVED/18REVIEW/11SYSTEM_ERROR.
  실패는 exact 원문 인용 불일치10개와 미완료 응답 검사1개다. 가드 제거·임의 재채점 없이
  원문/실패 사유를 보존하며 전체 감사의 완료율 및 유효 판정률을 별도로 보고한다.
  새로운 확정 정확도는 없고558/900은 불변이다. 모든102진단·Knowledge60은 이미 로컬 검증했다.
- GPU cutoff03:52:38 UTC와 최종회수04:22:38 UTC 유지; 결과 검증 전 Pod를 삭제하지 않는다.

## 2026-10-03 01:21 UTC /10:21 KST — Gemma 실제29검증 통과·900감사 시작

- 새 실제29 GPU smoke가29PASS/0FAIL이며 `full_audit_allowed=true`다. 원래 검증을 유지하고
  선택지 전수 모호성 검사를 거친 결과이며 실패 원문은 따로 보존했다.
- 저장된 원래 Qwen900응답 감사가8/900 진행 중, GPU100%/22304MiB 실제 연산을 확인했다.
  새 Qwen900 재추론이 아니라 frozen Gemma의 답 제출·파싱 감사다. 전체 완료/새 점수는 아직 아니다.
- 새CPU900v8의 모든payload/entry/hash/gold독립성·모델/원문/코드 identity·native17 검사를
  `cloud_gemma_cpu900_v8/local_receipt.json`에서 로컬 검증했다. 실제 통과 smoke는 닫힌 회수 준비 중이다.
- GPU cutoff03:52:38 UTC·최종회수04:22:38 UTC 유지. 완료 후 실제 원장/의미 검증·회수를 거쳐
  소유 Pod만 삭제한다. 기존558/900·지식60·공개 MMDL/GitHub는 불변이다.

## 2026-10-03 01:17 UTC /10:17 KST — Gemma의 선택지 모호성 안전검사

- 현재 비공개 protocol은 `gemma-frozen-final-audit-v6-exhaustive-choice-assessment`다.
  같은 gold-blind 호출에서 모든 보기를 평가하고, 하나만 지지되며 나머지가 모두 배제될 때만
  host가 선택을 확정한다. 단일 best-fit은 금지하며 native/host 모순을 차단한다.
- 49개 runtime/preflight+12개 controller CPU tests/Ruff/mypy3 PASS. 독립64조합/64모순/7잘못된
  witness 검사와 실제 Pod XGrammar17 schema/host 검사 PASS. 새CPU900v8는900COUNTED/overflow0,
  max50454·전체 raw 입력 유지이며 v6/v7과 token 원장이 byte동일하다.
- 원래27 실제 검증 계약은 바꾸지 않고 범주/세부범주2개를 추가해29 gate로 고정했다.
  controller7819/birth7298428을01:17:29 UTC 시작했다.29실제 검증 통과 전900감사 금지,
  03:52:38 UTC GPU cutoff와04:22:38 UTC 회수 기한 유지. 새GPU 성공은 아직 주장하지 않는다.
- 근거: `private_finish_v8_local_validation.json`, `private_finish_v8_launch_receipt_local.json`,
  `gemma_exhaustive_choice_independent_v1.json`; 이전 v7 실제26/27실패와 원문은
  `cloud_gemma_failed_smoke_v7/local_receipt.json`에 별도 검증·보존했다.
- 기존102진단/지식60/558/900 baseline·공개본은 불변이다. Pod삭제는 결과 검증 후의 별도 단계다.

## 2026-10-03 01:02 UTC /10:02 KST — 다분야 Knowledge Oracle 실제 완료

- 동일 pinned Qwen/BF16/vLLM0.11.0으로15분야60개 텍스트 시험을 완료했다.
  K1 30/30,K2 30/30·누락/시스템 오류/잘림/파싱 실패0. 실행 약2분35초,
  완료 요청 합110.966초·입력7660/출력4301토큰이며 설치/전체 과금 시간과 다르다.
- 원문60/입력60·record/prompt/request/seed/hash·고정 bank를 로컬/독립 검증했다.
  key는 로컬 후속 채점 전용이며 Pod에 업로드하지 않았다. 단순 기초 문제의 천장 결과,
  분야당2문항과 기존 I/O1의 비동일 문항/난도를 기록했다. DFA/그래프 풀이 오류도 별도로 남겼다.
- 근거: `cloud_knowledge_v1/local_receipt.json`, `knowledge_scores_v1/`,
  `knowledge_actual_independent_review_v1.json`,
  `claudedocs/Qwen3_VL_4B_Knowledge_Oracle_20261003.md`.
- Gemma v7 실제27검증은26PASS/1FAIL이므로900감사는 미시작이다. 실패는 넓은 표현을
  특정 세부 보기로 좁힌 의미 판정이며 guard를 완화하지 않는다. 새CPU900v7는900COUNTED/
  overflow0/max50454·원문 절단0, v6와 입력 원장 byte동일이다. 닫힌 자료 회수 진행 중이다.
- 기존102진단/558/900 baseline·MMDL/GitHub 불변. 실제 학습·공개 업로드·Pod삭제는 하지 않았다.

## 2026-10-03 00:40 UTC /09:40 KST — 다분야 Knowledge Oracle 추가 승인

- K1 지식/K2 상태→규칙 적용을15분야×각2문항(60호출)으로 추가 준비한다.
  `experiments/bottleneck_diagnostic/knowledge.py`는 기존 gold-blind runner와 최종답 형식을 재사용한다.
  Qwen revision/BF16·이미지 예산·sampling은 그대로이며 텍스트 시험은 이미지 없이 같은 VLM을 사용한다.
  별도 정답키는 추론에 접근시키지 않으며 최종4096·seed3407·03:52:38 UTC cutoff를 고정한다.
  K3는 기존 실제 I 응답을 재사용한다. 새 K1/K2는 더 단순한 자체 점검 문제라 K3와 동일 문항 통제 실험이 아니다.
  3합성 CPU tests/Ruff/mypy PASS; 실제 새 GPU 결과는 아직 없다. GitHub/MMDL 게시하지 않는다.
- Gemma v6 실제17GPU smoke는16PASS/1FAIL(고정 답의 의미 연결 보류)이므로900감사 금지다.
  원문 날조/복수 답 guard는 통과했고 실패는 별도 회수 원장으로 보존했다.
  `cloud_gemma_failed_smoke_v6/local_receipt.json`:33payload/17entry hash·source 연결 실제 검증.
  Pod00:30조회 GPU0%/1MiB·compute 없음; 기존 Qwen102/558/900와 P 수정본은 불변이다.

## 2026-10-03 00:18 UTC / 09:18 KST — Gemma 구조적 수정 검증·실제 재개

- 프롬프트 반복 변경 대신 추출 입력을 `question_type/raw_response/finish_reason`으로 분리했다.
  question/choices/gold/V8를 추출 호출에 주지 않는다. 보기 연결은 동결된 제출 답과 보기만 사용하는 별도 단계다.
  native JSON 분기에서 모호답의 null·정정 근거를 강제하고, 전체 답의 복수 후보/단위/부정과
  가정·인용·코드 fence를 공통 evidence 검사로 처리한다. 기존 V8 파서는 수정하지 않았다.
- stable runtime34/preflight12/controller12 CPU tests, Ruff·mypy3 PASS.
  독립 Sol의24가정/인용/코드표현·6기존 반례·5정상 대조 검증 PASS:
  `gemma_isolated_submission_independent_v3.json`. 실제 Pod XGrammar0.2.3의11native schema CPU 검사 PASS.
- 새 CPU900v6는900COUNTED/overflow0·max50454+1024=51478·원문 절단0이다.
  runtime SHA221c8f2b…/preflight873e854f…/controllerfc2b0bb7…를 별도13파일 archive로 비공개 전달했다.
  controller6363/birth6938159를 시작했고 실제 Gemma 서버가 로드 중이다. 아직17GPU smoke/900감사 완료가 아니다.
  GPU cutoff03:52:38 UTC·회수기한04:22:38 UTC를 유지하고 Qwen102는 재실행하지 않는다.
- P102원장 중80의 O0조건부 AI 검토를 독립 표본 검사하여7판정을4문항에서 수정했다.
  원래 v1은 보존하고 `p_structure_measurement_corrected_all30_v2/`를 별도 생성했다.
  38source hash·102결정·전체summary 재현을 `p_structure_measurement_corrected_main_check_v2.json`에서 확인했다.
  전체 지원1309/1877, 참조 회수710/1382이며 미확정530/452를 빼고 얻은 수치와 구분한다.
  사람/이미지 진실 인증·사전 중요관계·모듈 책임 분리는 미완료다. V8/공개 MMDL/GitHub 불변.

## 2026-10-02 23:44 UTC / 10-03 08:44 KST — 실제 GPU 중단 확인

- 사용자 상태 질문에 실제 장치를 확인했다. Pod `h447a4u6c0vq3k`는 켜져 있으나
  GPU0%/1MiB·compute process 없음이며 controller5916은 종료됐다. RUNNING을 추론 중으로 해석하지 않는다.
- v4는 실제13사례 중12통과/1실패였고, v5도11통과/2실패라900 GPU 감사는 시작하지 않았다.
  v5의 복수답은 AMBIGUOUS인데 extracted_answer가 null이 아닌 schema 실패다.
  Math_12는 실제 raw에 없는 선택지 문구를 evidence로 제출했다. literal guard가 차단했으며 완화하지 않는다.
  반복 프롬프트 수정만으로 같은 실패를 재실행하지 말고 원문/입력 경로와 모델의 실제 행동을 점검한다.
- v5 CPU900은900 COUNTED/overflow0·max50192+1024=51216이다. 닫힌19payload archive를
  실제 수신했으며 `cloud_gemma_cpu900_v5/`의 별도 byte 회수 검증을 진행한다. GPU 통과 증거가 아니다.
- P 의미 검토는80비교 가능 문항의 두40문항 AI 검토를 합쳐102원장(22비교불가)을 생성했다.
  `p_structure_measurement_reviewed_all30_v1/` 및 merged_source_check 원장에 기록했다.
  O0 조건부/사후 전체 항목 단위·사람/이미지 진실 인증 아님·중요관계 미측정을 유지한다.
  Energy_and_Power_20 reference uncertainty의 question-derived 문구 노출도 명시한다.
  Qwen102/749실호출과 원래558/900·공개 MMDL/GitHub는 불변이다. 전체회수/Pod삭제는 미완료다.

## 2026-10-02 23:23 UTC / 10-03 08:23 KST — Gemma 실제 smoke 실패 원인 재현·새 CPU gate

- v3의 실제13사례 smoke는10통과/3실패였다. 복수 후보의 revision evidence 누락,
  동일 금액·방향에서 통화 기호만 다른 표현의 의미 비교 실패, CPU/서버 renderer의
  메시지 형식 차이로 가장 긴 입력에1공백 token이 추가되는 원인을 구분했다.
  `cloud_gemma_failed_smoke_v3/local_receipt.json`으로 실패 원문·실제12completion을 회수했다.
- 새 judge prompt는 evidence 의무와 통화 기호의 한정된 표기 동치만 명확히 한다.
  validator·점수 비교·13사례 통과 기준은 완화하지 않았다. 서버의 지원 인자
  `--chat-template-content-format string`으로 CPU HF 문자열 경로와 맞췄다.
  BF16·모델·문맥131072·sampling은 유지하고 새 protocol과 별도 v4 소스/출력을 쓴다.
- local52 CPU tests/Ruff/mypy3/Bash PASS 및 독립 원인 검토 후 CPU900v4 실제검사 완료:
  900/900 COUNTED·overflow0·절단0, 최대50061입력(+1024=51085)이다.
  실제900 GPU 감사는 아직 미실행이며 다시13사례 smoke를 통과해야 시작한다.
- P 전체 의미 측정 큐는102개/80비교 가능·22비교 불가이며 gold 접근 없이 생성·검증했다.
  사후 전체 list-item 단위의 두 독립40문항 shard를 실제 AI 검토 중이다.
  아직 의미 정확도가 측정됐다고 쓰지 않으며 사전 중요관계 목록 부재도 유지한다.
  Qwen102문항·기존558/900·MMDL/GitHub는 불변이다.

## 2026-10-02 23:06 UTC / 10-03 08:06 KST — Gemma 새 gate 실제 실행

- runtime public 버전과 CUDA 휠 버전을 구분하되 BF16·exact wheel hash·HTTP 버전 검증은 유지했다.
  새 preflight11 CPU tests, controller12 CPU tests, Ruff·bounded mypy 및 별도 Sol 코드 검토 PASS.
- 새 CPU900v3는900/900 COUNTED·문맥 초과0이며 이전v2와900개 전체 token 원장 byte가 동일하다.
  바뀐 코드 hash에 새 manifest를 연결했다. 기존v2와 실패 기록은 덮어쓰지 않았다.
  실제24 payload파일을 암호화 회수하고 `cloud_gemma_cpu900_v3/local_receipt.json`으로 로컬 검증했다.
- 23:03:59 UTC PID4853/birth6497401의 controller v3를 별도 경로에서 시작했다.
  SHA46d94a4097f47a8ce21281f81ea76fe36db388f1ee75381a479bba07a5e53319;
  Qwen/coverage 재실행은 하지 않는다. 같은 Gemma BF16·131072 서버가23:05:07 READY,
  실제13사례 smoke를 실행 중이며 통과 전900감사는 금지한다. GPU cutoff03:52:38 UTC 유지.
- P 의미 측정은 원래 O0 기준과 실제 P를 gold 없이 연결하는 검토 큐를 구현 중이다.
  기존 서로 다른 단위의 AI leaf/group 검토를 합산하지 않으며 사전 중요관계 목록 부재를 숨기지 않는다.
  위 진행은900감사·P 의미지표·전체회수/삭제 완료가 아니다. GitHub/MMDL은 불변이다.

## 2026-10-02 22:50 UTC 근거 기준 — 추가42 완료·Gemma gate 수정

- 42coverage는22:29:44 UTC EXIT0,336조건/305실호출/20차단/11제외·누락/시스템오류0이다.
  닫힌23payload파일을 회수하고305입력/336원래strict-v1채점/전체summary를 독립 CPU 재현했다.
  `analysis_exports/bottleneck_20261003/cloud_coverage42_v1/`, `coverage42_closed_independent_check_v1.json`.
- pilot60+coverage42는102고유문항/30과목이며 CPU-v2의816행 집계는
  `export_all30_102_v2/`에 별도 저장했다. 실제749호출·47차단·20제외이며 기존558/900은 불변이다.
- Gemma v1은 FlashInfer sampler JIT의 nvcc부재로22:31실패했다. 실제설치소스의 지원flag0을
  별도controller에 고정해22:50서버READY를 확인했으나 smoke는 runtime0.24.0/휠0.24.0+cu129
  잘못된 동일성 검사에서 실제 judge 호출 전에 차단됐다. 실패·정리·기존CPU900v2는 보존한다.
  버전과 exactwheel hash를 분리하는 새preflight+새900 CPU gate를 검증중이며900GPU 감사는 미완료다.
- 독립 완료의무 점검은 P의 전체 의미구조 지표 미측정,LEGACY 실제재호출 없음,
  추가seed3408/3409·잔여160미실행 및 최종전체회수/Pod삭제를 분리한다. 미완료를 완료로 축소하지 않는다.
- 이 작업의 모든 코드·전사·결과는 내부용/Git제외다. MMDL/GitHub 동기화·push·학습은 하지 않았다.

## 2026-10-03 비공개 병목 진단 — 구현·입력 감사·로컬 점검

- 코드: `experiments/bottleneck_diagnostic/`, `experiments/gemma_verifier/`.
  Git 제외이며 공개 MMDL을 수정·동기화·push하지 않는다. V8 558/900과 원응답은 불변이다.
- 상태: 핵심 runtime/채점/예산/재개 IMPLEMENTED, 새 CPU64 tests/Ruff/mypy14 source files PASS.
  실제900 JSONL/지정 Parquet/982 이미지 RGB 픽셀 대조 완료; 질문·보기 참조 입력977회,
  해설 전용5회는 새 입력에서 제외했다. `analysis_exports/bottleneck_20261003/prepared/receipt.json`.
- 고정12 CPU dry-run 및 pinned processor I/T0/P 36조건의 실제 크기/grid/입력 토큰 감사 완료.
  마지막 Merger와 DeepStack3개를 실제 meta named_modules에서 확인했고 Vision과 분리했다.
  `module_inventory_final.json`을 정본으로 사용한다(이전 메타 목록은 tied weight/루트 라벨 정리 전 기록).
- Sol 전사24개를 원본 이미지로 별도 Sol이 검토했다:22 VALID/1 EXCLUDED/1 BLOCKED.
  사람 검증이 아니며 `AI_REVIEWED_NOT_HUMAN_VERIFIED`, 기본 human gate에서는0 VALID.
  `oracles/stress_reviewed.jsonl`, `oracles/review_receipt.json`에 근거·버전·hash를 보존한다.
- 로컬 BF16 CPU-offload 실추론은 Public_Health_14/Music_3 × I/T0/P/M/MI의10호출 완료,
  전부EOS·총1164출력 토큰·시스템 오류0이다. 중단된143/SIGTERM 시도는 보존하고 재개했다.
  두 P 응답의 image marker가 구조 schema와 달랐으므로 구조 추출 성공으로 인정하지 않는다.
  M/MI는 수정하지 않은 동일 P 응답을 공유했다. 소규모 기능 점검이며 성능 추정·RunPod 검증은 아니다.
  `local_smoke_2cases/completion_receipt.json`, `local_smoke_2cases_validated_scores/`에 근거를 보존했다.
  새 클라우드 P 프롬프트는 image ID 형식을 명시하므로 이 로컬 결과와 다른 prompt hash를 갖는다.
- Gemma는 저장900 요청의 gold-blind 준비와 실제 pinned tokenizer 전수 검사를 완료했다.
  checkpoint는 google/gemma-4-E4B-it, SHA ee0ef6023621cff504d758262d4e04895a5af4a2.
  원답의 근거 없는 선택지/단위·부호·modifier 탈락은 REVIEW로 막는 회귀 검사 포함.
  900/900 COUNTED·문맥 초과0, 최대 입력49,979/출력 예산 포함51,003 token이다.
  전체 응답을 자르지 않았으며 GPU 서버·13사례 smoke·900 감사·새 점수는 아직 미실행이다.
  `cloud_gemma_cpu900_v2/`에 실제 CPU 결과와 실행한 v5 소스를 회수했다.
  현재 로컬 preflight의 추가 타입 주석과는 byte가 다르므로 GPU gate는 불변 v5 소스를 사용한다.
- 10분 간격으로3090 용량을 확인한 뒤17:17:37 UTC에 단일3090 Pod `h447a4u6c0vq3k`를 생성했다.
  실제RTX3090/24576MiB·driver580.95.05·Python3.12·영구/workspace200GB를 확인했다.
  GPU $0.22/h, Network Volume 없음. SSH proxy는 SCP를 제공하지 않아 공식runpodctl의
  암호화 일회성 전송으로 비공개325,003,636bytes 묶음을 전송·원격 검증하고 실제 추론 중이다.
  `pod_creation_receipt.json`, `private_transfer_v1.tar.receipt.json`에 식별·해시를 보존한다.
  운영 기한은2026-10-03 04:22:38 UTC/13:22:38 KST; 파일럿 GPU4h gate 유지.
  60파일럿 후 사전262 표본 내 추가42문항으로 나머지21과목의 정/오답1개씩을 우선 점검한다.
  balanced 진단 정확도를 전체MMMU 정확도라 부르지 않는다. 추가 반복·확장은 실측 여유에 따른다.
- 독립 Sol 검토 전사:60파일럿은 O0 50승인/10차단, O1 44승인/7차단/9제외;
  추가42표본은 O0 35승인/7차단, O1 25승인/6차단/11제외다. 사람 검증으로 표시하지 않는다.
- 원본900/V8 채점 원장/공개MMDL의 불변성 재검증 PASS. 골드 접근 분리·원장 무결성·누락 조건 분모·
  절대 UTC 기한·실제 모델13파일 hash 및 backend 설정 identity 검사를 강화했다.
- 클라우드 초기 설치는 exact lock·pip check·실제 BF16 probe까지 통과했다. 모델 취득은
  컨테이너 기본 `HF_HUB_ENABLE_HF_TRANSFER=1`과 lock 내 선택 패키지 부재로 중단됐다.
  비공개 setup v3는 해당 값을0으로 고정하며, 이전 mount/환경 freeze/Pod 소유권을 검증한
  `--resume-download`로 기존 설치·실패 로그를 보존하고 취득만 재개한다. 추론 설정은 불변이다.
- 17:44:55 UTC: 지정 Qwen13파일/해시 확인 후 Pod 전용 PID888로12stress→60pilot 실행 셸을 시작했다.
  12stress는96조건 중94실호출 완료·1제외·1차단·시스템 오류0으로 연기 시험 gate를 통과했다.
  닫힌 결과를 회수하여 전96원장·채점·집계를 CPU 재현했다(`cloud_stress12_v1/local_receipt.json`).
  요청 시간1,558.674초·입력78,488/출력63,464 token이다. 고정 사례이며 전체 정확도 추정이 아니다.
  추가42coverage는 자동 시작하지 않고
  파일럿 실측 및 Gemma900 감사·회수 시간을 확보한 뒤 판단한다. 설치/모델 로드와 추론 시간은 분리한다.
- 18:34:03 UTC 확인:60pilot 진행 중,4문항의32조건 실호출 완료·시스템 오류0이다.
  Gemma 별도 native imports·pip check·7파일 다운로드/해시 검증 완료이며 GPU 서버/감사는 아직 미실행이다.
- 진단 인터페이스의 `FINAL_ANSWER: B. 16` 같은 유효 형식 누락을 확인했다. 별도 CPU 채점 v2는
  단일 문자·문자+정확한 해당 보기·고유한 보기 전문만 수용하고 gold/중간 풀이를 이용하지 않는다.
  12stress에서 추출19조건 변경(정확한 보기 형식16수용/문자·내용 충돌3미확정), 정오+9/−0조건이다.
  `cloud_stress12_v1/scoring_overlay_v2/`에 이전 채점과 함께 보존한다. 모델 향상·9개 새 정답 문항이 아니다.
  실행 중 Pod 프롬프트/코드·기존 V8는 변경하지 않았다. 새로운 진단 응답의 v8_score는 과거558/900과 다르다.
- 로컬 확장 선택기는60pilot+42coverage+160잔여=262 고유 표본·30과목을 확인했다.
  잔여160 전사320개도 Sol 독립 이미지 검토를 완료했다: O0 124승인/35차단/1제외,
  O1 106승인/27차단/27제외. 로컬 재검증은 모든 원문 이미지·전사 hash/O0 schema·고정 ID/기본 human gate 차단을
  확인했다(`expansion_oracle_validation_v1.json`). 아직 해당 잔여 표본의 GPU 추론은 시작하지 않았다.
- 로컬2문항 독립 감사는 Public Health의 표 label/count 보존과 Music의 note-position/ledger/sharp 누락을
  구분했다. 구조 schema 실패와 실제 내용 누락은 별개이며, 특정 내부 모듈의 원인으로 단정하지 않는다.
  `local_smoke_2cases_independent_audit.json`의16개 raw offset·2이미지 hash·P 공유 검증을 근거로 한다.
- 클라우드12개 P는 정상 stop/97–340 token/JSON schema 통과지만, 별도 원본 이미지 대조에서9문항의
  기호·위치·수치·관계 누락/오류를 관측했다. 다른3개도 구조 완전성을 인증한 것은 아니다.
  `stress_p_structure_review.json`의26원문 offset·12이미지 등장 hash·M/MI24개 P 공유를 로컬 재검증했다.
  `claudedocs/Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md`는 이를 담은 중간 보고서이며
  파일럿/Gemma/학습 완료나 내부 모듈의 유일한 원인 식별을 주장하지 않는다.
- 19:12:58 UTC 확인:파일럿119조건 중108실호출 완료/6차단/5제외·시스템 오류0이며 계속 실행 중이다.
  별도 비공개 continuation은 독립 재검토의 deadline/pidfd/실제900 reserve gate 수정 후
  로컬 및 Pod CPU/mock11 tests·Ruff·bounded mypy·dry-plan을 통과했다.
  15,091bytes의 별도 묶음과4개 코드 파일 hash를 원격 검증하고 PID2371로 시작했다.
  현재 `pilot_wait=WAITING`이며 원래PID888·v1 inference/v5 Gemma gate를 변경하지 않았다.
  파일럿 종료 후 실제 원장 검증→실측 여유에 따른42coverage→Gemma 실제13 smoke→900감사를 이어간다.
  controller GPU cutoff는03:52:38 UTC, 회수 여유30분이며 Pod 삭제는 자동화하지 않았다.
  `private_finish_v1_launch_receipt_local.json`, `finish_runpod_independent_review.json`에 근거를 보존한다.
  새CPU 검사는 합계75개(기존64+controller11)이며 GPU 완료·새 점수·로컬 전체회수는 아직 아니다.
- 19:21:40 UTC부터 별도 native nvidia-smi1초 sampler(PID2505, birth5163444)를 시작했다.
  이는 늦게 시작한 whole-device 표본이며 전체 run/allocator peak가 아니다. 모델·추론 코드는 불변이다.
  `device_sampler_v1_launch_receipt_local.json`에 scope·03:52:38UTC timeout·첫 표본을 기록했다.
  소유Pod의 실제 청구2 bucket과 hourly cost를 읽기 전용 확인했다(`owned_pod_billing_snapshot_v1.json`).
  최종 비용이나 현재까지의 전체 청구라고 주장하지 않는다. 계정/결제/다른 자원은 변경하지 않았다.
- 19:34:46 UTC 확인:60pilot는231조건(214실호출/9차단/8제외), 28문항의MI까지 종료·시스템 오류0이다.
  Qwen888/continuation2371/sampler2505의기존birth를 확인했고 continuation은계속pilot_wait=WAITING이다.
  기존 프로세스를 재시작하거나 입력·출력 예산을 바꾸지 않았다. 이는 중간 조회이며 파일럿 완료가 아니다.
- 19:45:33 UTC 확인:pilot279조건(259실호출/12차단/8제외),34문항의MI까지 종료·시스템 오류0이다.
  완료 요청 평균21.5095초·합계5,570.967초이며 wall time/남은 완료 시간의 확정값은 아니다.
  기존888/2371/2505 birth일치·continuation WAITING이다. 실행 중 추론은 변경하지 않았다.
- 비공개 exporter를 구현·독립 검토했다: 신규5회귀와 기존metrics2를 포함한7 CPU tests·Ruff·mypy2파일 PASS.
  중단된 실행 경로를 호출 수0으로 잘못 해석할 수 있는 집계 문제를 고쳐 실패 호출 수는 불확실/null,
  budget 예약 토큰은 관측 생성 토큰과 분리했다. 실제 닫힌stress12 export_v2의96채점행·groups/P집계·9입력 hash
  는 이전값과 일치한다. `cloud_stress12_v1/export_v2/`, `export_results_independent_review_v1.json`에 근거를 보존했다.
  새 테스트도Git 제외experiments하위에 두며 공개MMDL은clean이다. 실제Gemma replay/재시도/최종회수 완료 증거는 아니다.
- 20:06:59 UTC 확인:pilot308조건(285실호출/15차단/8제외),38문항의MI까지 종료·시스템 오류0이다.
  기존888/2371/2505 birth일치·continuation WAITING을 확인했고 추론을 재시작하지 않았다.
  완료 요청 평균은20:07:52 조회 시23.9407초이며 남은 시간/확장 완료의 확정값은 아니다.
- 비공개recovery는독립검토의입력경로범위R1을수정하고로컬/Pod CPU각4검사·Ruff·mypy2파일을통과했다.
  별도10,905bytes code bundle을암호화전송·원격5파일hash검증후 `/workspace/private_recovery_v1`에배치했다.
  실제로닫힌stress12의5파일을로컬압축/복원하고96원장·94raw hash불변을독립검증했다.
  실제Pod의완료된환경10파일+회수코드5파일도17,735bytes/SHA05432b4b...fd827로회수·추출검증했다.
  `cloud_environment_v1/recovery_receipt.json`, `recovery_stress_roundtrip_independent_v1.json`,
  `private_recovery_v1_deployment_receipt_local.json`이근거다. 가중치/cache/실행중pilot결과는미포함이다.
  byte회수receipt는실험전체완료/정답의미인증/Pod삭제승인이아니며원래v1/v5소스·공개MMDL은불변이다.
  `cloud_environment_independent_receipt_v1.json`은15파일·checkpoint/lock/기존metadata·도구5파일
  일치도독립확인했다. 설치기록검증이며가중치실물회수/새GPU준비성/추론peak인증은아니다.
- 20:26:43 UTC 확인:pilot329조건(306실호출/15차단/8제외),41문항MI까지종료·시스템오류0이다.
  기존888/2371 birth일치·continuation WAITING으로검증된대기이며추론설정을변경하지않았다.
  Sol의외부후보4개공식자료조사와main원문대조는
  `claudedocs/research_diagnostic_external_candidates_20261003.md`에만보관한다.
  과제/사용권/중복검사/관측과의조건부연결을구분했고최종순위·모듈효과·데이터clean인증은아니다.
  다운로드·학습·새cloud호출·MMDL/GitHub변경은하지않았다.
- 20:42:36 UTC 확인:pilot345조건(322실호출/15차단/8제외),43문항MI까지종료·시스템오류0이다.
  기존888/2371 birth일치·continuation WAITING이며실행설정은불변이다.
  완료P42개닫힌사본을실제Pod→로컬전송·record/raw/image byte검증했다.
  `pilot_p_shard_local_verification_v1.json`과gold-blind42packet에근거를둔다.
  새Sol의이미지대조·최종닫힌파일럿prefix연결은미완료다. 부분회수를전체실험/삭제인증으로쓰지않는다.
- 20:51:43 UTC 확인:pilot419조건(386실호출/24차단/9제외),52문항MI까지종료·시스템오류0·기존PID/birth일치다.
  별도Sol의P42개/47이미지gold-blind감사와main의원문/해시/1,051인용offset검증을완료했다.
  `pilot_p_semantic_review_v1.json`은오류/누락관측31·미확정7·확인범위내오류미관측4이며전수구조정답률이아니다.
  `pilot_p_semantic_review_local_verification_v1.json`에무방향선분endpoint역순의과잉오류판정수정도기록했다.
  최종닫힌파일럿prefix연결·전체파일럿/Gemma완료·최종회수/Pod삭제는아직미완료다.
- 21:04:14 UTC 확인:pilot471조건(435실호출/27차단/9제외),58문항MI까지종료·시스템오류0·원래PID/birth일치다.
  P경계5사례의별도Sol검토와main의원문/35인용/5이미지byte검증을완료했다.
  `pilot_p_review_boundary_check_v1.json`, `pilot_p_review_boundary_local_verification_v1.json`은
  header/start/down의schema모호성·C→c문자차이·ground기기분류를구분한다.
  기존42감사집계는불변이며31개를확정된시각실패수로사용하지않는다. GPU원장/기존V8/공개본도불변이다.


- 21:22:58 UTC 최신 확인:60pilot는480조건(444실호출/27차단/9제외),누락·시스템오류0으로 완료됐다.
  원래PID888은 종료했고 continuation2371의birth5109976은 일치한다.
  실제 닫힌32파일을1,690,254bytes/SHA03a9a523...db14e9로암호화회수했다(`cloud_pilot60_v1/`).
  원래source17/hash·444payload·480 strict-v1 채점·전체summary CPU정확재현과별도독립검증PASS다.
  `pilot60_local_experiment_verification_v1.json`, `pilot_closed_independent_check_v1.json`이근거다.
  기존P42사본의7,809,462bytes/334행prefix와모든42raw/record도최종원장과일치했다.
  현재로컬scoring.py/cli.py와원래소스의차이는보존·공개했으며v2를원래추론코드로표시하지않는다.
  별도CPU-v2 export의123추출변경·47정오증가/0감소는새모델추론/기존558점수개선이아니다.
  최종답384조건의NO_PARSE93에는length70이포함되고자연종료NO_PARSE23은별도gold-blind감사중이다.
  원래source namespace가회수도구패키지에가려졌던읽기전용collector첫시도는쓰기전에실패했다.
  cwd/PYTHONPATH를원래bundle로고정하고회수모듈만별도alias로읽어해결했으며추론코드를수정하지않았다.
  42coverage 실측예산gate는21:14:50 UTC에통과(예상12,574.724초,완료보장아님)했고다음초실제시작됐다.
  조회시20조건(19실호출/1제외)/2문항MI완료·오류0이며실제root는
  `/workspace/artifacts/private-finish-v1/coverage`다. 42전체/GemmaGPU/전체회수/Pod삭제는아직미완료다.

- 21:43:59 UTC 조회:42coverage145조건(131실호출/9차단/5제외)/18문항MI종료·오류0이다.
  controller2371/birth5109976와coverage3139/birth5842638은일치하며재시작/설정변경없다.
  `coverage_actual_progress_20261002T214359Z.json`은진행snapshot이고완료/Gemma/삭제증거가아니다.
  자연종료NO_PARSE23개gold-blind감사는명시적답변포기13/표현후보10을구분했고
  원문23개·53인용offset/sourcehash를검증했다(`pilot60_stop_no_parse_contract_local_verification_v1.json`).
  가정/문자내용충돌도보존하며추가정답계산/채점기수정은하지않았다.
  P나머지18개/21이미지감사와main의117assertion+16누락문맥+14경계인용검증도통과했다.
  `pilot_p_remaining18_local_verification_v1.json`에서Music11의정의되지않은전수전사관계와
  Physics18의기기해석/병렬연결오류를분리했다. 원래18감사/이전42감사는불변이다.
  60개P모두에AI검토가연결됐지만전체의미정답률/모듈책임/사람검증으로합산·인증하지않는다.
  보고서링크30개실존및기존V8/900원본/MMDL불변·Git제외·diff공백검사를확인했다.

- 22:03:43 UTC 조회:42coverage184조건(170실호출/9차단/5제외)/23문항MI종료·오류0이다.
  동일2371/3139 PID·birth·command hash를 직접 확인했으며재시작/설정변경없다.
  `coverage_actual_progress_20261002T220343Z.json`은진행snapshot이며완료/Gemma/삭제인증이아니다.
  닫힌60pilot의214조건쌍을종료/추출상태로연결했다(`pilot60_paired_finish_parse_controls_v1.json`).
  O0−I의11회복/3악화는자연종료·추출성공4/1,출력상한포함5/2,나머지형식·포기포함2/0으로구분된다.
  이는사후서술분해이며인과효과/새MMMU정확도/특정모듈실패율이아니다. O0는이미지대신전사이며IO0와다르다.
  방향/gold를제외한14변경사례packet의별도Sol검토와main의28원문/70인용/7상한말미검증도완료했다.
  `pilot60_O0_I_transition_review_local_verification_v1.json`과내부MD는Music10의동일B/형식변경,
  CS7의같은표/절차변경및PublicHealth9의이미있는수치/범주·합산·완결변화를구분한다.
  이감사는이미지를보지않은응답진술비교이며시각사실/모듈원인인증이아니다. 파서/점수/원문/공개본은불변이다.

- 22:24:11 UTC:실제coverage PID3139/birth5842638을최대55초범위에서반복확인하고live 상태를검증했다.
  289조건(265실호출/14차단/10제외)/36문항MI종료·시스템오류0이다.
  `coverage_actual_progress_20261002T222411Z.json`은닫힌행만집계한진행기록이며전체완료/회수/삭제인증이아니다.
  별도14변경사례AI감사원문·인용검증과보고서36개링크/Git제외/diff공백검사는통과했다.
  기존V8원장918파일/원래900JSONL/공개86파일불변검증도통과했고GemmaGPU는아직대기중이다.

## 2026-10-02 Gemma 4 로컬 LLM 보조 채점 계획

- `claudedocs/gemma4_mmmu_judge_plan_20261002.md`에 Qwen 고정 commit의 API 채점 경로와 Gemma 대체 계획을 기록했다.
  공개 Instruct 셸은 gpt-3.5-turbo-0125를 지정한다. 규칙 실패 시 응답을 선택지에 연결하고 Python에서 GT와 비교한다.
  공개 셸만으로 공식67.4% 내부 실행 이력까지 인증하지 않는다. 주관식 gold 선택지 변환·무작위 fallback은 구분했다.
- 권장안은 V8 보존 + Gemma 전수 감사이며 gold/V8 비공개 최종답 추출 뒤 별도 비교를 수행한다.
  공식 API 호출부만 교체하는 방식과 달리 규칙이 이미 인정한 철회 답도 검토하는 팀 자체 확장이다.
  E4B-it frozen/greedy/JSON 구조화 출력은 후보이며 revision·환경·성능은 구현 smoke 전 미확정이다.
- Deep Audit와 실제 원장 Art_22·Manage_20·Geography_4를 대조했다. Manage_20은 NO_PARSE가 아니라
  PARSED 이후 비교 손실이다. 실제 Gemma 판단·900 감사·새 점수는 아직 없다.
  문서만 추가했으며 V8/558정답(900문항)/원응답/HTML/MMDL/GitHub 불변, GPU·LLM API·학습·설치 미실행.

## 2026-09-30 P1 전체·오답 결과 PDF 변환

- `results/P1/V8_32K_highres.pdf`:900문항(정답558/오답342),5,943쪽,479,661,632bytes(479.66MB).
  `results/P1/V8_32K_highres_wrong_342.pdf`:오답342문항,4,071쪽,209,462,288bytes(209.46MB).
  모두512,000,000bytes 미만이다. 전체본은 무손실WebP HTML보다 커졌으며 PDF가 항상 더 작지는 않다.
- 본문에는 과목·질문·원래 보기·정답/추출답·종료/파싱 상태·이미지·모델 응답 전문을 포함한다.
  원응답과 같은897개 model_explanation은 중복 인쇄만 생략하고 다른3개와 gold_explanation은 별도 인쇄한다.
  모든 metadata·토큰ID·정확한 원본 문자열은 PDF 내부 `records.jsonl` 첨부에 원장 그대로 보존했다.
  PDF 본문은 줄바꿈/페이지 재배치·흑백 emoji 표시를 사용하며 비가시적FE0E/FE0F는 첨부 원문에만 유지한다.
  Math_12의 실제 대량 공백 출력도 보존했다. 과목/문항별 책갈피를 제공한다.
- 두 PDF의 모든 문항 본문을 공백 정규화 후 원문과 비교하고 이미지982/372회의 순서·크기·픽셀·페이지 경계를 검증했다.
  PNG alpha는 PDF soft mask와 합친 premultiplied RGBA로 대조했다. 이미지 축소/손실 압축은 하지 않았다.
  전체 첨부JSONL 원장 일치·기존 HTML SHA256 불변·복사 SHA256 일치 PASS.
  대표 페이지·emoji·음악 기호를 렌더링해 확인했고 내부 도구 Ruff/mypy 검사도 통과했다.
  독립 검토에서도900/342개 원장·책갈피 순서/페이지·30과목·이미지982/372회·파일hash 일치 PASS.
  검증 후 중복 중간PDF와 smoke PDF는 제거했고 최종P1 PDF와 소형 검증 증거는 보존했다.
- 기존 내부 HTML helper를 재사용하는 `analysis_exports/tools/export_v8_pdf.py`만 추가했다.
  PyMuPDF1.26.4는 변환 전용 임시 환경에서 사용했으며 평가 환경/lock에는 추가하지 않았다.
  증거: `analysis_exports/mmmu_20260924/p1_pdf_20260930/verification.json` 및 같은 폴더의 소형 화면 검증PNG.
  원본HTML/평가결과/파서/점수/MMDL/GitHub는 불변이며 GPU·재채점·외부API는 실행하지 않았다.

## 2026-09-30 V8 고해상도 과목별 공통점 내부 분석

- 사용자 요청으로 `claudedocs/MMMU_V8_70pct_subject_analysis_20260930.md`에 분석을 저장했다.
  Finance를 포함한 ≥70% 12과목과 <70% 18과목을 비교했다. 원장900개와 지정 revision Parquet의
  ID·질문·보기·정답·유형 일치, Parquet manifest hash 및 전체/오답 HTML 기존 hash 일치를 검증했다.
- 정답은558/900 그대로다. Hard 비율10.00%/26.85%, length1.67%/15.56%, NO_PARSE2.50%/16.30%이며,
  자연 종료·추출 성공만 비교해도81.38%/61.14%다. 과목별 탐색적 상관과 Music 제외 민감도를 함께 기록했다.
- Marketing의 표23/23·Public Health의 표23/25, 기계공학 투상도/단면 선택0/8, 병리/농업의
  length·NO_PARSE0 등 서로 다른 양상을 확인했다. 대표 이미지·응답 검토이며 전900개 독립 의미 감사는 아니다.
  전문지식 보유량·학습 노출·원인별 점수 기여도를 측정했다고 주장하지 않는다.
- 동일V8 이전32K554→고해상도558은 +61/−57, 순증가4개다. 해상도 단독 통제 실험으로 해석하지 않는다.
- 근거: `analysis_exports/mmmu_20260924/v8_subject_groups_20260930.json`.
  재검증: `PYTHONDONTWRITEBYTECODE=1 /home/jang2/mmdl-env/bin/python analysis_exports/tools/analyze_v8_subject_groups.py`.
  내부 스크립트 Ruff/mypy·900건 일치·보고서30과목표 대조 PASS. 기존 설치 pyarrow와 표준 라이브러리만 사용했다.
  원본/HTML/평가 설정/파서/점수/MMDL/GitHub는 불변이며 GPU·외부 API·학습은 실행하지 않았다.

## 2026-09-29 P1 전체 HTML을512MB 미만으로 축소

- `results/P1/V8_32K_highres.html`:554,470,586→398,995,534bytes(약399.00MB/380.51MiB,28.04% 감소).
  보수적인512,000,000bytes 기준보다113,004,466bytes 작다.
- 기존 내부 생성기에서 원본 RGB PNG를 무손실 WebP로 변환하되 더 작을 때만 채택한다.
  이번 파일의982개 이미지 표시는 모두 WebP이며959고유 원본의 크기·모드·모든 픽셀 동일성을 검사했다.
  해상도 축소·손실 압축·풀이/metadata 생략·모델 재실행·재채점 없음.
  `data-sha256`은 원본 PNG 해시이며 변환된 WebP byte의 해시는 아니라는 설명을 HTML에 포함했다.
- 900고유 문항·정답558/오답342·이미지982회·전체 텍스트/metadata 원본 일치 검증 PASS.
  별도 스트리밍 HTML 검사에서도900개 순서/원문/metadata·982개 RGB 픽셀/크기/매핑·외부 자산 없음 확인.
  내부 코드 Ruff/mypy PASS. 현재 SHA256:`cc918d40263364d00205c0e8f1aabc6eee973d5d06bd53c08601d6a005f759c6`.
- 이전 PNG 내장본은 `analysis_exports/mmmu_20260924/p1_png_backup_20260929/V8_32K_highres.html`에 보존했다.
  오답 전용342문항 HTML은 byte 불변이며 원본 결과·다른 P1 파일·MMDL/GitHub는 변경하지 않았다.
- 증거:`analysis_exports/mmmu_20260924/p1_compact_images_20260929.json`.
  Windows Chrome headless에서 실제 축소본의 이미지/문항 표시 확인:
  `analysis_exports/mmmu_20260924/p1_compact_full_preview.png`. 임시 브라우저 프로필은 정리했다.
  재검증:`/home/jang2/mmdl-env/bin/python analysis_exports/tools/export_v8_wrong_html.py --verify-only`.

## 2026-09-29 P1 전체·오답 HTML에 이미지 내장

- `V8_32K_highres.html`:900문항(정답558/오답342), 이미지982회/959고유PNG, 554,470,586bytes.
  `V8_32K_highres_wrong_342.html`:오답342문항만, 이미지372회/369고유PNG, 270,314,786bytes.
- `source.json`이 가리키는 기존 추론 run의 원본 크기 RGB PNG를 축소·재압축 없이 base64로 내장했다.
  모델 내부에서 resize된 입력 텐서의 복제라고 표현하지 않는다. 질문·선택지의 모든 이미지 참조를 포함한다.
- 생성기 및 독립 HTMLParser 검사에서 문항ID 전체집합·순서·정오·이미지번호/크기·원본PNG byte/해시·풀이/metadata 무변조 검증 PASS.
  내부 도구의 Ruff/mypy PASS; Windows Chrome headless로 두 실제 HTML의 이미지 표시 확인.
  분야/정답/질문은 기본 표시, 긴 풀이 접기와 원본 크기 이미지 보기 제공. 외부 자산/JavaScript 필요 없음.
- 기존 두 텍스트본은 `analysis_exports/mmmu_20260924/p1_text_only_backup_20260929/`로 백업한 뒤
  검증된 새 파일로 교체했다. 다른 P1 파일·평가 원장·MMDL은 변경하지 않았고 GitHub 게시는 하지 않았다.
- 증거: `analysis_exports/mmmu_20260924/p1_embedded_images_20260929.json`,
  `p1_images_full_preview.png`, `p1_images_wrong_preview.png`. 임시 생성 폴더와 Chrome 검사용 프로필은 정리했다.
  재검증 명령: `python3 analysis_exports/tools/export_v8_wrong_html.py --verify-only`.

## 2026-09-28 내부용 V8 오답 HTML

- `results/P1/V8_32K_highres_wrong_342.html`: V8 고해상도32K의900문항 중 정답558(62%)/오답342(38%)에서
  `correct=false`342개만 포함했다. NO_PARSE97/답추출후불일치245, length85/stop257, length∩NO_PARSE76.
- 생성기와 독립 검사에서342고유ID·오답집합 완전일치·정답혼입0·raw_response/metadata 원본 일치를 확인했다.
  기존 HTML의 문항 section을 그대로 보존한 텍스트 전용 파일이며, 이미지 픽셀은 원본 자료에 별도 보관한다.
- 크기82,874,884bytes; SHA256 `ea06ed1b417a13a15f9a786f6dd9fd8cd70fecdd0b79b2afebe47bf530f19274`.
  내부 도구 `analysis_exports/tools/export_v8_wrong_html.py --verify-only`로 재검증 가능. Ruff/mypy PASS.
  브라우저 화면 검사는 Chrome 부재로 미실행; 파일 구조·원장·내용 검증은 통과했다.
- HTML/도구는 모두 Git 제외 경로다. MMDL/원격에 반영하지 않았고 기존 P1 파일과 원본900개는 보존했다.

## 2026-09-28 보고서 시간 표기 개선

- GitHub 직접 수정 `fd8f582`를 먼저 동기화하여 보존했다. 시간은 평가24시간52분41초,
  모델로드28초, 전체호출24시간53분29초로 표기했다(초 단위 반올림, 설치·다운로드 제외).
- 보고서 생성기의 기존 시간 표시 함수도 변경하고 24시간 초과·반올림 경계·미제공 및 표 출력 회귀를 검사했다.
  제출본171 CPU tests/Ruff/mypy26/Bash·공개파일/staged 검사 PASS. 원본 측정 JSON과 seed·평가·채점은 불변이다.
- 게시 `197838c11557911a14a46d87b7f52fda0cac5b3c`; MMDL/원격 HEAD 일치·clean 확인.
  GPU 실행 없음. 문항별 seed 정책의 목적은 순서/재개에 독립적인 seed 배정이며, 출력 동일성이나 정확도 향상 보장은 아니다.

## 2026-09-26 P1에 V8 HTML 추가

- `analysis_exports/mmmu_20260924/team_final_v8/`의 세 `review.html`을 `results/P1/`에 복사했다.
  `V8_2K.html`23,401,205bytes, `V8_32K.html`94,808,607bytes,
  `V8_32K_highres.html`113,326,660bytes. Windows 위치는 `C:\projects\multimodel\results\P1`이다.
- 원본/사본 `cmp` 통과. 각900고유 문항·정답478/554/558, 외부 자산 의존 없는 텍스트 HTML 확인.
  기존 V6/V7 HTML6개·설명MD1개는 보존하여 현재 P1은 HTML9개+MD1개다.
  Git 제외 확인. 채점/원문 변경·GPU 실행·GitHub 게시 없음.

## 2026-09-26 기타 특이사항·한계 교체

- 사용자 요청으로 보고서 §8의 기존 본문을 제거하고 동일 출력32768에서 이미지 입력 확대의 작은 점수 차이
  (554→558/900,+0.44%p), 파서 표현 범위, 모호한 답의 미확정 처리, 독립 검증의 한계로 교체했다.
  전처리·문맥 한도·환경도 달라 해상도만의 인과효과로 단정하지 않았다.
- 공개 변경은 보고서 한 파일의 §8뿐이며 §1–7(격차 분석934자)·팀 정보·점수·코드는 불변이다.
  보고서9 CPU tests/Ruff/mypy25/Bash·공개파일 전체/staged·diff 검사 통과.
- 게시 `6b019dfd696dc4b029b1bb83802d7b2a80362a6c`; 원격 default/MMDL HEAD 일치·clean,
  상위 allowlist/공개 checkout·보고서 byte 일치 확인. GPU·LLM·재채점 미실행.
- 호환용 REPORT_STATUS 표식은 빈 채로 남겼다. 기존 생성기를 다시 실행하면 상태 설명이 추가되고
  §7이 일반 진단으로 바뀌므로 수동 작성한 §7–8을 재검토해야 한다. 생성기는 변경하지 않았다.

## 2026-09-26 GitHub 팀 정보 동기화·근거 중심 격차 분석

- 사용자 승인으로 원격의 두 보고서 변경을 MMDL에 fast-forward한 뒤 상위 정본에 반영했다.
  동기화 기준 `ac451fa706da8d3b991a5d50a19c5b0e02fd0f51`; 팀명 Team 2,
  팀원 장우석·정예찬·김기원·김정현을 그대로 보존했다.
- 이후 공개 변경은 `reports/mmmu_baseline.md` §7뿐이다. 공백·줄바꿈·Markdown 링크 포함934자이며
  채점 표현 손실(Manage_20), 반복 후 미완료(Accounting_9), 겹침을 고려한 오류 집계,
  seed·이미지 입력 비교의 한계로 격차 원인을 추론했다. 다른 절·30과목 표·점수는 불변이다.
- 저장 V8 원장에서 558/900, 이전32K554/900, NO_PARSE97/length90/겹침76,
  오답342의 상호배타적 구분85/21/218/18 및 두 사례를 읽기 전용으로 확인했다.
- 제출본 `bash scripts/test_reference.sh`:170 CPU tests/Ruff/mypy25/Bash 통과.
  `scripts.check_submission` 전체·staged 검사, `git diff --check`, §7 분량·그 외 본문 불변 검사 통과.
- 게시 완료 `1b3e51bb47cf43629b23d3f072e78fac121adcbf`, default `feat/mmmu-baseline`.
  원격 HEAD와 MMDL HEAD 일치·clean, `scripts.sync_submission --check` 및 상위/공개 보고서 byte 일치 확인.
  평가 코드·파서·설정·저장 결과는 변경하지 않았으며 GPU/RunPod/LLM/재채점은 실행하지 않았다.
- §7은 이번 결과에 맞춘 수동 분석이다. 기존 보고서 생성기는 실행 결과로 일반 진단을 재생성하므로,
  향후 생성 명령을 다시 실행한 뒤에는 §7을 검토해야 한다. 이번에는 생성기 코드를 변경하지 않았다.

## 2026-09-25 제출 템플릿 정리·MMDL 재현 경로 확인

- 승인된 보고서 초안을 `reports/mmmu_baseline.md`에 반영: 템플릿8절, 실제 프롬프트·설정·출처·자체 설계 이유,
  30과목 표·558/900(62.00%)·1000자 이내 격차 분석. 내부 버전 개발사/원장 설명은 본문에서 제외한다.
- 생성기가 옛 장황한 설명을 복구하지 않게 수정하고, 정적 본문과 실제 평가 설정이 다른 입력을 거부한다.
  실측은 원본 GPU 기록에서, 점수는 검증된 V8 채점에서 가져오며 그 차이를 한 문장으로 공개한다.
- README에 프로젝트 지도와 데이터 manifest만 있고 모델 cache가 없는 환경의 명시적 준비 명령을 추가했다.
  lock·다운로드·loader·프롬프트·이미지·vLLM·채점·출력 의존 경로에 필요한 공개 파일을 확인했다.
- 상위176/공개본170 CPU tests·Ruff·mypy25·Bash·공개 파일 검사 PASS. 원본900/V8 overlay로 생성한 보고서 byte 일치,
  8절/30과목/558정답/상대 링크 확인. 합성 예제로 초기 파서의 짧은 답 누락·후행 보기 오인·주관식 부분일치도 확인했다.
- 평가 코드·파서·설정·기존 결과는 변경하지 않았다. 새로운 의존성 설치와 GPU/4090 재평가는 미실행이다.
- 게시 완료: `fa98910d560b196396395010bd29d7a6d00446ce`, default `feat/mmmu-baseline`.
  상위 allowlist/MMDL/commit의86파일 byte 일치, 원격 HEAD 일치, clean 확인.
  상위/MMDL의 실제 파일권한도 일치한다. WSL 마운트의0777 표시와 Git mode는 별개이며 셸은 `bash`로 실행한다.
  게시 증거: `analysis_exports/template_publication_20260925.json`.

## 2026-09-24 보고서 정본 경로 통일

- 사용자 요청: 공개 보고서 정본을 `reports/mmmu_baseline.md`로 통일하고 root `Assignment_1.md`는 제외.
  제공된 SUBMISSION_TEMPLATE의8개 항목을 유지하며 README/assignment 안내/생성 명령도 같은 경로를 사용한다.
- 생성기는 reports의 본문을 읽고 지정 출력 하나만 작성한다. 자동 루트 복제/사본 동기화를 제거했다.
  상위 Assignment_1.md는 과거 기록으로 보존하지만 공개 allowlist나 현재 생성기에서 사용하지 않는다.
- README에 clone/commit 확인/환경 설치/고정 버전 다운로드/평가/CPU 재채점/보고서 경로를 연결했다.
  필요한 소스·lock·config·manifest·공식 참고 코드/라이선스 존재를 읽기 전용으로 점검했다.
  기존 raw 응답은 외부 보존이며 새 GPU/4090 end-to-end 실행은 이번에도 하지 않는다.
- 게시 완료: `53a37bcdad221f88ce5ba355f37e1b78e744d3e5` (default `feat/mmmu-baseline`).
  86파일의 상위 allowlist/MMDL/커밋 byte 일치, 원격 HEAD 일치, clean checkout 확인.
  제출본168 CPU tests/Ruff/mypy25/Bash, 보고서/게시 도구 회귀13개, 링크/템플릿8절 검사 PASS.
  기존 B2 원본900 + V8 overlay로 실제 보고서 재생성 후 현재 reports 본문과 byte 일치 확인.
  원장·생성·채점 점수는 바꾸지 않았다. 삭제한 공개 루트 사본은
  `analysis_exports/submission_backups/20260924T142715Z/Assignment_1.md` 및 Git 이력에서 복구 가능하다.
  게시 receipt: `analysis_exports/report_path_publication_20260924.json`.

## 2026-09-24 V8 완료·공개본 분리

- V8 세900 CPU 재채점 완료: 478/554/558정답(53.11/61.56/62.00%). B2 제출 baseline 유지.
- 원본/비채점 필드/소스 snapshot/hash·2700 gold 독립 추출·재생·산술 검증 통과.
  pipeline167 CPU tests/Ruff/mypy25/Bash PASS. 완전 의미 판정이나 독립 holdout 인증은 아니다.
- V6→V8 +4/−11. qualifier3/선언충돌4/모호·잘림충돌4의 감소 한계와 기존 회귀39/46·2/2·4/4를 문서화했다.
- 결과/HTML: `analysis_exports/mmmu_20260924/team_final_v8/`. 내부 상세:
  [V8 완료 기록](../claudedocs/mmmu_v8_pipeline_20260924.md).
- 새 GPU/4090 검증/LLM/학습/본격 모델 오답 분석 없음.
- 사용자 요청으로 제출용 `MMDL/` 독립 Git clone을 만들었다. `scripts/sync_submission.py` allowlist로
  코드·설정·lock·작은 결과·README·Assignment만 복사하며 AGENTS/docs/claudedocs/원본자료는 제외한다.
  상위 AGENTS에 이후 게시 규칙을 기록했다. 기존 원격 공개 파일은 회수 가능한 로컬 백업에 보존한다.
- GitHub 게시 완료 SHA `88dd56bb6887369196c510a43c1462215eb2fe2b`.
  원격 default `feat/mmmu-baseline`과 MMDL HEAD 일치·clean status 확인. 허용 목록87파일.
  상위173/제출본167 CPU tests, Ruff/mypy25/Bash/공개 파일/문서 링크/보고서 사본/score-time CSV/hash 검사 PASS.
  변경 전 공개 파일은 `analysis_exports/submission_backups/20260924T141534Z/`와 Git 이력으로 복구 가능하다.
  게시 receipt: `analysis_exports/submission_publication_20260924.json`.

## 2026-09-24 P1 최종 구성 — 설명문 1개 + HTML 6개

- 사용자 요청으로 P1의 `V6_V7_code.zip`을 삭제하고 `V6_V7_implementation_ko.md`로 대체했다.
  실제 보존 코드 기준의 추출 순서·비교 규칙·실패 처리·점수·감사 한계를 설명한다.
- HTML 6개는 변경하지 않았다. P1은 Markdown 1개 + HTML 6개이며 ZIP이나 하위 폴더는 없다.
  원본 코드 ZIP 두 개는 `analysis_exports/mmmu_20260924/p1_full_archive/code/`에 남아 복구 가능하다.
- 아래 ZIP 구성 기록과 `scripts/export_p1_analysis.py`는 이전 요청의 산출물이다.
  해당 스크립트의 ZIP 포함 검증/생성 명령은 현재 Markdown 구성에 적용하지 않는다.

## HISTORICAL — 2026-09-24 P1 구성 (ZIP 1개 + HTML 6개)

- 사용자 요청에 따라 `results/P1/`에는 `V6_V7_code.zip`과
  `V6_2K.html`, `V6_32K.html`, `V6_32K_highres.html`,
  `V7_2K.html`, `V7_32K.html`, `V7_32K_highres.html`만 둔다.
- ZIP 내부는 `V6/`, `V7/` 코드이며 결과물을 포함하지 않는다. HTML은 각각900문항의
  질문·보기·응답 전문·정답·채점 정보를 유지하되 이미지 픽셀과 외부 탐색 링크를 제외했다.
- 기존 전체 묶음은 삭제하지 않고 `analysis_exports/mmmu_20260924/p1_full_archive/`로 이동했다.
  아래 파일 수·크기·이미지 검증 기록은 이동 전 전체 묶음에 대한 과거 기록이다.
- `scripts/export_p1_analysis.py --verify-only`: 정확히7개 파일, ZIP 원본 코드 일치,
  HTML900문항 및 원문·채점 정보 보존 검사 통과. Ruff와 `git diff --check` 통과.

## HISTORICAL — 2026-09-24 P1 전체 분석 묶음 (현재 별도 보존)

- `results/P1/` (Windows `C:\projects\multimodel\results\P1`): V6/V7 당시 코드 ZIP 2개,
  각 버전×세 실행의900문항 결과와 해당 점수로 새로 렌더링한 HTML 6개를 분리 보관했다.
- 공유 PNG959개, 전체2700응답의 중복 메타데이터를 줄인 JSONL, 판정 변경64건 전문,
  비교 CSV·감사 ledger·과제/출처 문서·웹 분석 요청문을 포함한다. 원본 응답은 자르지 않았다.
- 총1,341,919,222bytes,6466파일(manifest 포함). Linux staging의6465파일 hash를
  Windows 복사본과 대조했다. 각HTML900 sections/982 image references, PNG959개 검사 완료.
- 코드ZIP은 별도 폴더의 파서 검사 V6 51개·V7 69개 통과. source/archive 및 원문 hash 보존.
  당시 생성기는 보존 폴더의 `export_full_layout.py`에 남겼다. P1은 `.gitignore`로 공개 추적에서 제외했다.
  원본·현재 파서·모델 설정은 변경하지 않았고 웹 업로드/GPU/학습/commit/push는 하지 않았다.

## 2026-09-24 최신 상태 — V7 거부, V6 runtime 복원

| 실행 | 현재 V6 점수 | V7 시도 점수 (거부됨) |
|---|---:|---:|
| baseline_2048 | 479/900 (53.22%) | 463/900 (51.44%) |
| continuous_32k | 558/900 (62.00%) | 539/900 (59.89%) |
| official_b2 | 560/900 (62.22%) | 539/900 (59.89%) |

- V7 implementation/module/tests 및 runtime selection을 제거하고 기존 V6 runtime을 복원했다.
  현재 V7 source/command는 사용할 수 없다. 원본 V6 소스 hash와 동일함을 확인했다.
- V7 시도는 총64개 판정을 바꿨다: V6 정답→V7 오답60개, 반대4개.
  AI 보조 원문 검토는 손실60개를 명확한 V7 false negative46, 엄격 문자열 정책 누락2,
  모호/잘림12로 분류했고, V7이 회수한4개는 정상으로 확인했다.
- 합성 회귀에서는 V6의 부정/복수 후보 및 문자열 부분일치 false-positive 예가 확인됐다.
  실제 전체 데이터 false-positive가 0이라고 보장할 수는 없다.
- 최신 공개 집계: `results/scoring-v6-20260924/summary.json`.
  재현 V6 결과: `analysis_exports/mmmu_20260924/v6_v7_audit/v6_replayed/`.
  거부된 V7 실행/후보/공개 결과/snapshot은 `analysis_exports/mmmu_20260924/rejected_v7/`에 보존했다.
- 상세 감사: [V6/V7 false-positive 및 점수 차이](../claudedocs/mmmu_v6_v7_false_positive_audit_20260924.md).
  V7 정책 초안은 [historical/rejected 기록](../claudedocs/mmmu_v7_scoring_20260924.md)이다.
- 132 tests 통과. FROZEN V4 inference protocol과 generation settings는 변경하지 않았다.
  GPU 실행, Git commit/push는 하지 않았다.

## REJECTED / HISTORICAL — 2026-09-24 V7 세 실행 CPU 재채점

> 아래는 거부된 V7 시도의 재현 가능한 과거 기록으로 남긴다. V7은 runtime에서 제거됐으며
> 아래 점수/절차를 현재 채점이나 현재 사용 가능한 명령으로 해석하지 않는다.

| 실행 | V6 | V7 | V6 대비 정답 증가/감소 | 미확정(900개 분모에 포함) |
|---|---:|---:|---:|---:|
| baseline_2048 | 479/900 | 463/900 (51.44%) | +1/−17 | 249 |
| continuous_32k | 558/900 | 539/900 (59.89%) | +1/−20 | 114 |
| official_b2 | 560/900 | 539/900 (59.89%) | +2/−23 | 129 |

- 원본900×3의 raw/input/gold/finish_reason와 모든 비채점 필드 불변, ID900고유·30과목×30,
  record hash·samples 일치·독립 재채점·과목 산술·source hash 검증 완료. V4/V6도 각각2700개 재현.
- 자체 최종답 추출과 명시적 비교 규칙. 숫자에만 pinned MMMU `normalize_str`의 두 자리 반올림 사용.
  정답을 추출에 전달하지 않으며 LLM/random/legacy fallback/단위 환산/재생성 없음.
- 모든150 tests/Ruff/mypy25 source files/Bash 검사 통과. 추가 화학식·문단 정정 결함은 합성 회귀와
  독립 검토로 수정했다. 엄격한 정책으로 실제 답 표현도 누락하므로 모든 점수 감소를 거짓 정답 제거라고 부르지 않는다.
- 저장: `analysis_exports/mmmu_20260924/team_final_v7/`, 실행별 summary/CSV/predictions/samples/source,
  루트 `v6_to_v7_comparison.json`과 정확한 `scorer_source.tar.gz`.
  이미지·HTML은 원본에 남으며 기존 viewer는 V7 점수로 바뀌지 않았다.
- 공개용 작은 집계: `results/scoring-v7-20260924/`.
  [정책·근거·검증·제한](../claudedocs/mmmu_v7_scoring_20260924.md);
  [제출 보고서](../Assignment_1.md)와 `reports/mmmu_baseline.md` 동기화.
- 개발 후보1(456/530/530)·후보2(463/538/537)는 `team_final_v7_candidate_1/`·`team_final_v7_candidate_2/`에
  이동 보존했다. 삭제 없음. V7 개발에 validation 사례를 참고했으므로 별도 holdout 검증으로 표시하지 않는다.
- 당시 V7 작업본은 로컬 미커밋이었다. scoring hash `0db83f5dfc8912e4163502d5b9b89e1a80b967f39c1819ead371ee7ca66838cf`.
  아래 V5 B2 561점은 역사적 후보값이고 V6는560점이다. 이번 작업에서 GPU/RunPod/학습/commit/push 없음.

## 2026-09-24 V5 파서 수정·두 32K 실행 CPU 재채점 완료

- 사용자 승인 후 `team-final-answer-v5` 추가: 답 선언의 순서/가정문 후보 필터를 수정했다.
  V4 기본 동작과 기존 inference protocol/원본 결과는 유지하며 V5는 명시적으로 선택한다.
- 이전 R2는 V4 556 → V5 558/900(62.00%, +2/−0), B2는 557 → 561/900(62.33%, +5/−1).
  B2의 명백한 최종답 누락 3개를 회수했다. 추가 2개 정답은 length에서 명시 답 추출이며,
  Electronics_25의 최종 75 V를 gold 50으로 잘못 인정했던 1개는 오답으로 바로잡았다.
- `analysis_exports/mmmu_20260924/team_final_v5/`에 두 결과/양방향 비교/소스 스냅샷을 보존했다.
  두 원응답 파일 hash와 모든 비채점 필드 불변, 1,800개 record hash/독립 재채점/coverage 검증 완료.
  전체 126 tests, Ruff, mypy 24 source files, Bash 검사 통과. V4 저장 출력 1,800개 재현.
- 상세: [점수 진단과 후속 수정](../claudedocs/mmmu_b2_score_diagnosis_20260924.md).
  GPU 재추론/유료 작업/LLM judge/코드 commit·push는 하지 않았다.

## 2026-09-24 R2/B2 점수 차이 CPU 진단

- [상세 진단](../claudedocs/mmmu_b2_score_diagnosis_20260924.md): 같은 감사 V2는 R2 584 → B2 575,
  같은 확정 V4는 R2 556 → B2 557이다. 과거 V2 64.89%와 B2 V4 61.89%를 직접 성능 하락으로 비교하지 않는다.
- 이미지 입력은 실제 확대됐고 900개 CPU C/저장 입력/엔진 prompt IDs 대조에 불일치가 없다.
  V4 정답 60개 증가·59개 감소, length 75 → 90으로 정확도 개선은 입증되지 않았다.
- V4가 앞의 correct-answer 후보 때문에 뒤의 명확한 Answer를 놓치는 결함을 최소 예제로 재현했다.
  B2 Chemistry_24/Mechanical_Engineering_29/Physics_25의 명시적 정답 누락을 확인했다.
  기존 parser tests 30개는 통과했다. 진단만 수행했으며 코드/채점 정책/원본 결과 수정이나 새 추론은 하지 않았다.

## 2026-09-24 새900 로컬 회수·검증·Pod 삭제 완료

- `mmdl-official-b2-20260923-evaluation`:900/900,30과목×30,557정답(61.89%),누락/시스템 오류0.
  실제 추론 commit `6d6d4a1a4bed1785e8770902d65e12b54340ba75`, protocol `mmmu-val-official-vllm-b2-v2`,
  scorer `team-final-answer-v4`. 평가89,561.304초(약24시간53분). 원래 R1/R2 결과는 변경하지 않았다.
- 사용자 요청대로 `analysis_exports/mmmu_20260924/official_b2_complete/`에 결과를 회수했다.
  archive351,474,793bytes(335.19MiB), manifest2810파일·합698,625,051bytes.
  SHA256 `cf848eb87c124c835a182f541e155bcec2e723ed15bc6e81e3e021395b389c5b`가 원격·sidecar·로컬에서 일치한다.
- 별도 고정 commit checkout으로 안전 추출·파일/레코드/코드/input hash·900고유ID·과목별 산술·
  raw 재채점·이미지959개 디코딩·viewer 참조를 확인했다. `extracted/local_receipt.json`은 `LOCAL_VERIFIED`다.
  bundle/cleanup 안전 검사12개 통과. 추가 smoke3개 레코드와 참조 이미지 hash도 검증했다.
  Playwright MCP의 Chrome 부재는 설치된 Windows Chrome으로 대체해 로컬 결과·이미지 렌더링을 확인했다.
- 2026-09-24 07:44:32 UTC Pod `58nm0qrpgwc4ub` 삭제API204, 이후조회404와 동명 목록0을 확인했다.
  30GB container/80GB Pod workspace는 삭제되어 복구 불가지만 결과는 로컬에 보존된다. 별도 Network Volume 없음.
  정확한 증거는 회수 폴더의 `recovery_receipt.json`·`resource_ledger.json`에 있다.
- 이번 요청 범위는 결과 회수와 Pod 삭제다. 제출 보고서 표 재작성·GitHub 최종 결과 게시·추가 유료 실행은 하지 않았다.
  아래2026-09-23 실행중 표기는 역사 기록이며, 현재 클라우드 작업 상태가 아니다.

## 2026-09-23 최신 상태: 채점/입력 통제 후 신규3090 평가 승인

- 기존 R1/R2 모두900완료·회수·독립검증·Pod삭제 완료. 원래427/900과459/900은 불변 기록이다.
- `docs/P0/MMMU_audit`의 원본 V2 code/hash/8tests 확인 후 두900을 실제 CPU 재채점:
  536/900(+128−19),584/900(+148−23). 문항별 응답hash·추출·판정 모두 감사표와 일치한다.
- V2는 length/open 기존 판정 유지 한계가 있어 별도 `team-final-answer-v4`를 확정했다.
  일반 설명을 답 선언으로 취급한 후보 버그를 합성 회귀 검사로 수정하고, open은 최종 구간에만 공식 evaluator를 적용한다.
  두900 전체 재채점은479/900·556/900이며 원래 결과와 감사V2를 덮어쓰지 않는다.
  CPU111 tests/Ruff/mypy24source/Bash 검사 통과. 새 입력·모델 GPU 통제 실험18개를 완료했다.
- CPU900 입력 대조 완료: A는 기존 token수/grid/tensorhash 모두 일치. A→B897개,
  A→C900개 텐서 차이, C최대5627입력. 출력32768 유지, 전체context40960으로 고정한다.
- 게시 commit `a7fb5fec8d6b2a96c29e4577b351926e1aaba4b9`를 Pod `pdszzbcroakws1`에서
  clean checkout했다. RTX3090 24576MiB, driver580.65.06, Python3.12.3이며,
  GPU API 단가$0.22/h에 storage는 별도다. 기존 사용자 비용 상한 철회 범위를 적용한다.
  exact lock 설치·pip check·CUDA/BF16 doctor·지정 model/MMMU 다운로드를 통과했다.
  05:55 UTC A/B/C 각6/6·시스템 오류0을 완료하고 guard를 통과했다. 각2정답·length1이며
  평가 시간은 A1259.523s/B1346.895s/C1188.180s다. 작은 진단 표본으로 정확도 개선을 주장하지 않는다.
  C smoke 후05:57 UTC 동시1 full900을 시작했으나 사용자 요청으로 중단했다.
- **06:14:18 UTC Pod 삭제 완료:** `pdszzbcroakws1` 삭제204·후속 조회404.
  중단된 `mmdl-input-v2-full-evaluation` 부분 결과는 사용자 요청대로 회수하지 않고 Pod 디스크와 함께 폐기했다.
  기존 완료 R1/R2는 그대로 보존했다. 완료 A/B/C만 `analysis_exports/mmmu_20260923/new_baseline/`
  `mmdl-input-v2-controls.tar.gz`(3,400,152 bytes)로 회수했고 로컬18 record hash를 확인했다.
- 새 동시2 protocol `configs/eval/mmmu_val_official_vllm_b2_v2.yaml`은 C와 ID·batch만 다르다.
  기존 independent continuous refill을 재사용한다. 이미지/출력/문맥/seed/scorer 조건 완화는 없다.
  CPU111 tests/Ruff/mypy24source/Bash 통과. GitHub 게시 후 새3090의3문제 smoke→새900을 실행한다.
  CUDA Graph/async 추가 가속은 미검증 후보이며 이번 실행에 켜지 않는다. 전체 가속률은 아직 미확정이다.
  승인 조건/출처/명령은 [EVALUATION_V2](EVALUATION_V2.md)에 기록한다. 아래9/22의 실행중 표기는 역사 기록이다.
- **06:33 UTC 새900 진행 확인:** commit `6d6d4a1a4bed1785e8770902d65e12b54340ba75`,
  Pod `58nm0qrpgwc4ub`, RTX3090 24576MiB, driver580.126.09, Python3.12.3, API$0.22/h+storage.
  같은 pinned base image의30GB container/80GB workspace에서 GitHub clean clone→exact lock→pip check→
  CUDA/BF16→지정 다운로드→동시2 GPU smoke3/3·오류0을 통과했다. 추론 시점의 코드 SHA는 이후 문서 commit과 구분한다.
  smoke 평가30.717559s, 출력1078tokens,500ms 장치 전체 메모리 관측 최대23,707,254,784bytes(약22.08GiB).
  과제 점수가 아닌 기능 검사이며 full900 무오류/2배 가속을 보장하지 않는다. KV cache79376tokens.
  새 `mmdl-official-b2-20260923-evaluation`은06:31:57 UTC 시작해06:33:13 UTC3/900, 진행 중2개, 실패기록0이다.
  `/workspace/launcher.log`와 `/workspace/artifacts/jobs/mmdl-official-b2-20260923.008-evaluation.log`에서 확인한다.

2026-09-22. 최신 운영 문서: 로컬 smoke/부분 검증 → 팀 GitHub 고정 commit의 clean clone →
RunPod 단일 `mmmu-val` evaluation 900 추론 → 로컬 회수·검증 → 전용 자원 삭제.
새 분석용/테스트용 full run은 만들지 않으며, 저장된 evaluation 결과를 점수와 실패 검토에 함께 사용한다.
기존 A/B run은 역사 자료다. 단, 이미 실행된 continuous32k legacy run이 900개와 독립검증을 완료하면 canonical score 근거로 채택할 수 있다.
팀 저장소는 `https://github.com/jang2296/MMDL`, 최초 총 RunPod 비용 상한은 USD 6.80이었다.
현재 상한 철회는 아래 최신 승인 기록을 따른다. 이 문서 갱신만으로 기존 Pod를 중지·삭제·재배포하지 않는다.
대여 직전 GPU+storage 실단가에서 회수/복구 여유를 제외한 최대시간을 계산·기록한다.

## 단일 평가 전환 및 회수 승인 (2026-09-22)

- 새 CLI는 `--suite mmmu-val`, `<job>-evaluation` 하나만 실행한다. smoke는 별도이며 두 번째 full900은 없다.
- 기존 `assignment`/`analysis`의 원본 run ID·commit·hash를 보존한다. 이미 실행 중인 continuous32k 결과는 COMPLETE900·독립 재채점·무결성 검증 후 그대로 제출과 실패 검토에 쓸 수 있다.
- legacy 재개는 원래 commit에서만 한다. 보고서 생성 코드 commit과 실제 추론 commit은 다를 수 있으며 이를 숨기지 않는다.
- CPU 76 tests, Ruff, mypy 22 source files, Bash syntax, 단일평가 dry-run 통과. 새 이름의 GPU900은 재실행하지 않았다.
- 사용자는 먼저 완료되는 Pod를 로컬 회수·검증 후 삭제하도록 승인했다. 14:14 UTC 기존2048 launcher만 SIGSTOP하여 불필요한 후속900을 차단했고, 당시 evaluator는 그대로 완료시켰다.

### 기존 2,048-token 결과 회수·Pod 삭제 완료

- 기존 `mmdl-val-20260922-3090-assignment`는 900/900, 30과목×30개, 정답 427개, 미해결 시스템 오류 0으로 완료됐다. 이는 2,048-token reference의 결과이며 canonical 32k 성적표에 섞지 않는다.
- 실제 추론 commit은 `af961a36b2471e978e96efd7347e150cb0bbce00`이다. 같은 commit의 로컬 checkout에서 900개 고유 ID·원문 재채점·입력/레코드/코드 해시·PNG 959개 디코딩·HTML 이미지 링크·과목별 산술을 독립 검증했다.
- 2026-09-22 14:36:02 UTC 로컬 검증 완료. 저장 위치는 `$MMDL_ARTIFACT_ROOT/recovered/baseline-2048-complete-20260922/`이며, `results.tar.gz`, `extracted/artifacts/`, `recovery_receipt.json`을 보존한다.
- 압축 파일은 332,612,462 bytes(317.20 MiB), 원본 artifact 1,927개는 376,203,674 bytes(358.78 MiB)다. archive SHA-256은 `11cba0dff07a228c7ee70332f3b10859b66894a9349687507017c5072a3444e6`이다. 모델 가중치·전체 HF cache·venv는 포함하지 않는다.
- 14:36:23 UTC 전용 Pod `2j4knzy36jtzmv` 삭제 API 204, 이후 조회 404 및 Pod 목록 부재를 확인했다. Pod의 60 GB 작업 디스크는 삭제되었고 결과는 로컬에 보존된다. 별도 Network Volume은 없었다. 새 `vf8wla9q5v9gm8`는 RUNNING으로 유지했다.
- 원래 A/B job metadata는 변경하지 않았다. 두 번째 full900을 실행하지 않았으므로 원래 paired bundle 전체를 검증했다고 하지 않고, 단일 완료 run의 별도 recovery manifest/receipt로 회수 증거를 남겼다.

아래의 A/B 순서·대기 상태는 당시 실행 기록이다. 별도 B900을 시작하라는 현재 지시가 아니며 위 단일평가·회수 승인으로 대체되었다.

## 과거 실행 전환 — continuous scheduling (2026-09-22)

사용자 승인에 따라 기존 reference Pod는 유지하고, 새 분석 Pod의 partial 결과를 보존한 뒤
일시 중지하여 장문 생성 원인을 점검했다. 새 분석 run은 70/900, 시스템 실패 0으로 보존되었고
그 중 61개는 EOS, 9개는 `finish_reason=length`(각 32,768 token)였다. 9개가 전체 생성
399,758 token의 294,912 token(73.77%)을 차지했다. 예시 장문은 Architecture_and_Engineering
1에서 반복 문장이 약335회, 15에서 동일 유형 문장이 약79회 관측되었으나, 이는 원인 가설의
근거이지 모델 일반 동작의 증명은 아니다. 이후 사용자 요청으로 해당 분석 Pod와 로컬
70문항 결과 디렉터리·압축 백업을 삭제했다. 수치는 삭제 전 관측이며 새 실행에 재사용하지 않는다.

새 protocol 설정은 `configs/eval/mmmu_val_continuous_vllm_v1.yaml`의
`mmmu-val-fast-vllm-32k-continuous-v1`이다. vLLM 0.11.0 `LLMEngine.add_request()`/`step()`으로
최대 2개 request를 유지하고, 하나가 끝나면 최종 row를 즉시 저장한 뒤 다음 독립 문제를 refill한다.
`async_scheduling=false`, eager, BF16, P0, seed/image/parser, 출력 상한 32,768은 유지한다.
고정 batch protocol `mmmu-val-fast-vllm-32k-v1`은 역사 기록으로 보존하며 결과를 섞지 않는다.
구현 commit `702ed44c4c45f454746ec4b371ad1b7403b67856`을 GitHub에 게시했다.
새 RTX 3090/24GB Pod가 같은 commit을 clean checkout하고 lock 설치·CUDA/BF16·고정 다운로드·
3문제 GPU SMOKE를 통과했다. 분석 900문제를 새로 시작했으며 첫 확인은 5/900, 시스템 오류 0이었다.
이전 reference Pod는 그대로 유지했다. [상세 검증](../claudedocs/continuous_vllm_20260922.md).

이번 새 배포에 한해 사용자가 비용을 더 지불할 수 있다고 명시적으로 승인하여
`MMDL_COST_POLICY=user_waived` opt-in을 사용한다. 이는 결제/자동충전 변경이나 비용 0을
의미하지 않으며, storage reserve·실제 단가·자원 식별·회수/정리 기록은 계속 요구한다.
기본 strict 비용 정책은 유지한다. 이번 두 Pod의 기존 비용 제한 watchdog은 사용자 상한
철회에 따라 해제했으며, 새 배포도 자동 중지 예산을 임의로 만들어 기록하지 않는다.

새 continuous engine의 CPU mock 검사는 완료 row를 다음 request보다 먼저 보존하고,
완료 순서가 입력 순서와 달라도 기록하며, 후속 오류가 앞선 완료 결과를 지우지 않는지 통과했다.
전체 72 tests, Ruff, mypy 22 source files, Bash syntax, staged 공개 파일 검사가 통과했다.
설치된 vLLM 0.11.0 API/실제 SamplingParams의 CPU 검사도 통과했으며 로컬 GPU 추론은 추가하지 않았다.
Accounting_1, Agriculture_1, Biology_29를 `max_new_tokens=32768`로 실행했다.
모두 EOS 종료했으며 출력은 각각 303/5/415 tokens였다. 정답 1/3은 기능 SMOKE의 관측일 뿐
최종 성능이 아니다. 평가 구간 17.1465 s, 500 ms GPU 전체 메모리 관측 최대 20.6025 GiB였다.
Agriculture 결과 저장 후 Biology가 시작됐을 때 Accounting은 미완료여서 실제 rolling refill을 확인했다.

연속 실행의 `generation_seconds`는 각 engine `step()` wall time을 당시 active request 수로
나눈 attribution 값이고, 실제 요청 지연은 `request_latency_seconds`로 별도 기록한다.
따라서 request 시간 합을 전체 elapsed time으로 해석하지 않으며, throughput은 engine invocation
wall time으로 산출한다. 32,768은 Qwen 공식 MMMU 평가 source의 `out_seq_length` 값이며,
모델 card의 VL 16,384와 다른 평가 recipe 값이다. 어떤 설정도 자동으로 낮추지 않는다.

## 2026-09-22 가속 전환 — 진행 중

### 게시 및 별도3090 실행 기록

- 가속 코드/보고서 게시 SHA: `34c5cdb5559e00d1bda92e4932914344d112aba4`.
  GitHub feature ref의 SHA 일치 확인. 최종 CPU 단위검사61개, Ruff, mypy(22 source files),
  shell syntax와 staged 공개 파일 검사를 통과했다. 로컬 추가 모델 추론은 하지 않았다.
- 새 분석용3090은 GitHub에서 위 commit을 clean clone했다. 2026-09-22 02:26 UTC 부근
  `mmmu-val-analysis` bootstrap을 시작했다. GPU3090 24576MiB, driver555.42.02,
  Python3.12.3, `/workspace` XFS 영구 mount, 설치 전 CUDA/UVM 검사 PASS.
  별도 exact vLLM lock 설치→doctor→고정 model/MMMU 다운로드→2문제 GPU smoke→900 순서다.
  이 기록 시점에는 설치 중이며 vLLM GPU 추론 성공이나900 완료로 보고하지 않는다.
- 해당 base image의 SSH 환경에는 `RUNPOD_POD_ID`가 없어, provider MCP에서 확인한
  실제 Pod ID를 실행 shell에서 명시했다. 비밀키를 환경·저장소에 복사하지 않았다.
- 비용은 이전 failed Pod와 기존/새3090을 합산한다. 두3090 GPU 실단가는 각각$0.22/h이며,
  storage를 별도 보수적으로 합산한12:00 UTC 중지 안전장치가 Pod 생성 전에 설정되었다.
  그 시각까지 보수적 합산 상한$5.20, 회수·정리 여유$1.60이며 총$6.80을 넘기는 승인이 아니다.
  billing API는 지연되므로 이미 청구된 금액만으로 잔액을 추정하지 않는다.
- 기존3090 읽기 전용 재점검(02:24:55 UTC):118/900, 시스템 실패0,
  EOS80/length38,103,823 생성tokens/6,641.079초, GPU33%, VRAM19,696/24,576MiB.
  기존 실행/코드/설정은 바꾸지 않았다. 분석900 검증·회수 전 기존 Pod를 삭제하지 않는다.

이 절은 아래 이전 실행 순서보다 우선한다. 사용자는 기존 느린3090 유지,
가속 구현→추가 로컬1문제→GitHub→별도3090 분석900→검증·회수 후 기존Pod 정리→
새3090 제출900을 승인했다. 두 Pod가 잠시 겹쳐도 총 비용은 USD6.80 이내다.

- 기존 `mmmu-val-v1` 설정과 실행 중인 Pod/코드는 보존한다.
- 새 `mmmu-val-fast-transformers-v1`: batch1, 자동 SDPA, 매토큰 CPU streamer 없음.
- 새 `mmmu-val-fast-vllm-32k-v1`: batch2, 요청별 seed, BF16, native generated-only presence penalty,
  prefix cache 없음. 모델/processor revision, 데이터900, P0, sampling, 이미지 범위, parser는 유지한다.
  사용자가 잘림 문제를 지적하고 공식 생성 상한32768을 명시 승인했다. 전체 문맥36864와 구별한다.
- vLLM 전용 exact lock과 CUDA/UVM 설치 전 probe, single-role suite/bundle을 구현했다.
  로컬 격리 환경의157개 패키지 호환 검사와 vLLM0.11.0 import/CPU processor 검사가 통과했다.
  사용자 최신 지시에 따라 로컬 추가 모델 추론은 하지 않고 새3090에서 GPU 검증한다.
- vLLM 내부 입력 token IDs와 pinned processor reference IDs를 비교한다. reference pixel tensor hash를
  실제 엔진 내부 tensor 검증으로 표현하지 않는다. 동시처리 시간은 batch walltime과 배분 시간을 구별한다.
- 추가 로컬1문제 완료: `smoke-fast-local-one`, 312 tokens/EOS, 생성174.740초/전체199.549초,
  오류0. peak VRAM allocated5,043,425,792/reserved5,093,982,208 bytes, RSS11,221,233,664 bytes.
  이 실행의2048 상한은 새32768 승인 전 조건이다. 새3090 vLLM/32768 검증을 대체하지 않는다.
  출력 길이가 달라졌으므로 이전 실행 대비 전체시간 비율을 순수 가속 배율로 주장하지 않는다.
  기존34개 로컬 결과는 재실행하지 않았다. 새3090 smoke/900은 아직 미실행이다.
- 기존 Pod 읽기 전용 점검(2026-09-22 01:48 UTC 부근):75/900, 시스템 실패0,
  EOS50/length25,69,505 생성tokens/4,449.426초. GPU40%, VRAM19,696/24,576MiB,59도.
  입력에 따라 메모리는 증가하지만 이 점검에서 OOM은 없었다. 길이 제한의 품질 제약은 존재한다.
- 확인 순서: CPU 검사 → 로컬1문제 → 게시/합산 비용 watchdog → 새3090 analysis900 →
  로컬 완전 검증/열람 → 기존 partial 회수/정리 → 새3090 assignment900 → 회수/최종 정리.

## 순서와 증거

1. 발견/공식 자료 확인: 완료. 초기 폴더는 AGENTS.md와 과제 지침·템플릿
   3개뿐이며 Git 저장소·기존 코드·local_sources는 없었다.
   루트의 실제 과제 원문을 읽고 hash 기록 후 local_sources로 이동·ignore했다.
   feat/mmmu-baseline 로컬 브랜치 생성, commit/push 없음.
2. 환경/다운로드: 완료. Python 3.12.13, RTX 5060 Laptop 8151 MiB,
   Windows driver 592.82, WSL available RAM 약 21 GiB 확인.
   torch 2.8.0+cu128/CUDA12.8, transformers4.57.1, sm_120 및 BF16 matmul PASS.
   증거: 외부 setup/doctor.json 및 setup/*install.log, env/requirements-eval.lock.
   WSL ext4 available 655 GiB, 실제 VHDX 호스트 C: 잔여 378,914,590,720 bytes.
   다운로드 전 추정 peak와 별도로 50 GiB 이상 운영 여유를 유지한다.
   기본 Hugging Face cache는 비어 있고 지정 mmdl 경로는 없었다.
3. 평가 구현/CPU 검사: 평가 CLI·P0·채점·이미지·저장·재개·열람본 구현.
   22개 CPU 검사 PASS, lint PASS, typecheck PASS. 실제 2-image processor tensor/grid 검사 PASS.
   모델 두 shard와 MMMU validation 30파일, MMMU-Pro 10-way test 2파일 해시 검증 완료.
   최초 Hugging Face CDN DNS 오류는 자체 회복, 기존 다운로드 프로세스 이어받기 확인.
4. 1문제 → 1과목 30문제 → 최소 다중이미지/주관식 smoke: 첫 1문제·30문제 완료/검증 PASS.
   실행: `bash scripts/eval.sh --run-id smoke-one --mode smoke --limit 1 --no-download`.
   실제 BF16 모델 로드 2.464초. GPU에 vision/embedding/head와 decoder16개,
   나머지 decoder20개 RAM offload. 첫 응답 2,048 tokens / length 종료,
   생성 1,237.489초, 전체 1,260.610초, peak VRAM reserved 5,460,983,808 bytes,
   allocated 5,399,902,720 bytes, process RAM RSS 11,315,146,752 bytes.
   SMOKE 1/1 완료, 시스템 실패 0. 길이 상한에 도달했으므로 30문제 잘림률 실측 전
   초안 길이를 확정하지 않는다. 점수에 따른 설정 변경은 하지 않는다.
   외부 `runs/smoke-one/backend.json`, `setup/smoke-one.log` 참조.
   완료 전 해당 프로세스 swap 0 bytes 확인. 원문 JSONL/이미지/hash/HTML 보존 검증.
   독립 사후 검사: effective generation 설정, 실제 token→raw decode, 입력 정수형/hash,
   이미지 hash, JSONL 동등성, public artifact 참조 hash 모두 PASS.
   증거: 외부 `setup/smoke-one-audit.json` (1문제만 검증한 증거; 900문제 완료 증거 아님).
   다음 실행: `bash scripts/eval.sh --run-id smoke-accounting --mode partial --subject Accounting --no-download`.
   2026-09-21 18:01 UTC 완료 확인: Accounting 30/30, EOS 15/length 15, 시스템 실패·누락 0.
   프로세스 exit0. 전체 22,222.849초, 생성 포함 평가 22,194.080초, 37,848 생성 tokens.
   peak VRAM reserved 5,733,613,568 / allocated 5,433,335,296 bytes, RAM RSS 11,303,710,720 bytes.
   입력 tensor 전체 CPU 재계산/hash·P0/chat template·PNG decode/hash·token→raw decode·seed·
   raw 재채점·JSONL·public 참조/hash·escaped HTML 구조를 30개 모두 검증했다.
   통합 전 실행 code_files 28개 원본 일치. 증거: 외부 `setup/smoke-accounting-audit.json`.
   PARTIAL 18/30은 제출용 점수가 아니다. 추가 모델 추론 없이 사후 검증했으며 재실행하지 않는다.
   50%가 2,048 token 상한에 도달했다. 이 제한을 숨기거나 길이가 충분하다고 간주하지 않는다.
   중복된 첫 샘플의 seed/messages/input hash/generated token IDs/raw response/parsed answer가
   최초 smoke와 모두 일치. 현재 형식의 샘플 자체 hash도 검증 완료.
   증거: 외부 `setup/smoke-repeat-audit.json`; 같은 GPU의 1문제 관측에 한정한다.
   데이터 실제 coverage 900 고유 ID, MCQ847/open53. 이미지 개수 1:857/2:24/3:5/4:8/5:6.
   원본 README를 직접 로드하면 미취득 dev/test metadata를 요구하는
   ExpectedMoreSplitsError가 발생하여 검증한 과목별 원본 Parquet를 name=subject로 로드.
   검증을 끄지 않았으며 30×30/ID/hash 강제 검사 통과.
   증거: 외부 setup/data-coverage.json, setup/data-check.log.
5. 공통 protocol: 로컬30+구조3 검증 후 FROZEN. 변경은 상태 표기뿐이며 생성/이미지/P0/채점은 동일하다.
   2,048 token 상한은 실측된 잘림 제한을 명시한 팀 예산이다. 충분한 길이나 공식 지정값으로 주장하지 않는다.
   GitHub feature 게시 완료: `13b44e4c0f768d84f8c4d153b323371ff7168217`.
   원격 refs/heads/feat/mmmu-baseline의 SHA 일치 확인. 실제 RunPod 실행 SHA는 이후 수정 시 다시 고정한다.
   공개 대상 86개 파일의 실제 index bytes 비밀키/개인경로/크기 검사 PASS.
   개인 .serena 설정과 별도 방법론 문서는 제외·보존했다. 자체 코드의 whitespace 검사 PASS;
   vendored 공식 소스의 기존 공백 경고는 원본 hash 보존을 위해 수정하지 않았다.
   origin fetch=팀 HTTPS / push=동일 팀 SSH. 초기 빈 저장소의 feature 브랜치에 첫 게시를 완료했다.
   기존 SSH 인증에서 jang2296 계정 확인. 별도 GitHub 플러그인 설치는 필요하지 않다.
   공개키에 대응하는 기존 RSA 키로 비대화형 인증 성공, 비밀키 권한 0600 확인.
   Pod에는 공개키만 전달했다. 실제 SSH 실행·파일 쓰기·SCP 회수와 파일 해시 검증을 완료했다.
6. RunPod A/B: A 실행 중 / B 대기. 이전 실행 SHA `08e00ca802bb25f6915464df1f29ceb17b7eaa7e` 게시 확인.
   작업 전 Pod 0 / Network Volume 0. 5090 전용 Pod `0l1rrpy7m41lkq` 생성 성공(2026-09-21 19:12 UTC).
   Network Volume 없음. 60GB Pod volume/30GB container. 실제 GPU 단가 $0.69/h.
   독립 STOP watchdog을 먼저 가동했다. CUDA 진단 실패 후 조기 STOP/EXITED 확인, 감시 해제.
   SSH·SCP용 직접 TCP 포트 연결과 실제 5090 32607MiB/driver580.178.04/Python3.12.3,
   /workspace 영구 XFS mount 60GiB 확인. GitHub exact SHA 직접 clone·격리 lock 설치·pip check 성공.
   doctor는 CUDA 초기화 오류로 실패했다. 직접 libcuda cuInit(0)=999, /dev/nvidia-uvm open=EIO(5).
   CUDA control/GPU 장치는 열렸고 libcuda와 kernel driver 버전은 일치한다. 호스트 UVM 계층 장애이며
   정확한 커널 원인은 미확정이다. wheel/모델/평가 조건을 바꾸지 않았고 원격 추론은 0회다.
   진단/설치 로그 9개 SCP 회수·원격/로컬 SHA256 전부 일치. 외부 recovery/<job>-<pod>/에 보존.
   모델·benchmark 다운로드 전 실패였으므로 900개 결과는 없다. 진단 회수를 FULL LOCAL_VERIFIED로 쓰지 않는다.
   유일한 진단 사본이 로컬 검증된 뒤 장애 Pod를 삭제했다. 2026-09-21 19:27 UTC 조회:
   계정 Pod 0 / Network Volume 0. 실패한 준비 작업의 자원 정리만 완료; 전체 평가 완료는 아니다.
   Community 4090 생성 요청 2회와 교체용 EU-RO-1 5090 요청 1회가 재고 부족 HTTP400으로 거절됐다.
   최초 재고 부족3회 후 같은 조건의 재시도를 중단했다. 19:31 UTC 새 US-IL-1 4090 표시가 나타나
   공급 조건 변경을 근거로 해당 위치에만 1회 추가 요청했지만 역시 HTTP400이었다(재고 거절 누적4회).
   표시 재고와 실제 배정의 차단이 지속된다. 19:33 UTC Pod 0 / Network Volume 0 재확인.
   당시 목표를 BLOCKED로 전환했다. 2026-09-22 사용자가 GitHub 직접 취득 실행을 재지시하고
   4090 또는 3090 사용을 명시적으로 허용했다. 팀명/팀원은 미제공을 유지한다.
   재고 API의 LOW 표시와 실제 배정 가능 여부는 다르다. 동일 조건의 무근거 재시도는 하지 않는다.
   목표가 허용한 동급 이상 단일 GPU 대안으로 5090 32GB의 명시적 hardware/doctor guard를 추가했다.
   CPU 단위검사 41개·Ruff·mypy·shell syntax PASS. 평가 조건 변경이나 추가 로컬 추론은 없다.
   원래 운영 상한7.5시간 전에 중단했다. 청구 조회에서 이전 Pod GPU+디스크 합계 $0.1447284467 확인.
   재개 전 Pod/Network Volume 모두0 확인. 3090은 표시 재고 MEDIUM, Community $0.22/h;
   3090 명시적 profile/실제 장비 검사를 추가하되 공통 평가 조건과 dependency lock은 바꾸지 않는다.
   3090 추가 후 CPU42/Ruff/mypy/shell syntax 및 FROZEN/GPU-only profile 검사 PASS. 추가 로컬 추론 없음.
   새 게시 SHA `af961a36b2471e978e96efd7347e150cb0bbce00` 원격 일치 확인.
   2026-09-22 00:14 UTC Community 3090 Pod `2j4knzy36jtzmv` 배정, 실제 GPU $0.22/h.
   container30GB/volume60GB, 별도 Network Volume 없음. 외부 STOP 2026-09-22 22:14:12 UTC 가동.
   새 실행22시간+보수적 디스크 비용+이전 청구 합계 약$5.271, 회수 여유 약$1.529.
   배정 호스트 CUDA12.7이 cu128 컨테이너의 cuda>=12.8 조건을 만족하지 못해 부팅이 거부됐다.
   Pod STOP/EXITED 확인 후 공식 base 이미지 digest로 교체하여 00:26 UTC 재시작했다.
   `runpod/base@sha256:a5aead56b5ed7754235250afface107a8a19646ac34f62c87e5f41eb147fa7b2`.
   SSH에서 실제3090 24,576MiB/driver565.57.01/Python3.12.3/영구 XFS volume60GiB 확인.
   libcuda cuInit(0)=0 및 UVM 장치 열기 PASS. GitHub af961a3 직접 clone, exact lock 설치131.440초,
   pip check/FROZEN config/doctor BF16 PASS, 고정 모델·MMMU 다운로드45.835초 완료.
   실행 job=mmdl-val-20260922-3090. 사전 smoke1/1 완료/시스템 실패0, 실제 모델 GPU-only device map.
   smoke 생성50.679초/전체80.206초, peak VRAM allocated9,162,626,560/reserved9,275,703,296 bytes.
   00:33 UTC 제출용 A900 실제 평가 시작. 완료 후 별도 B900→검증/포장 순차 실행이다.
   외부 STOP deadline은 그대로 유지한다. 사전1문제는 전체900 결과나 시간 추정의 충분한 표본이 아니다.
   검사를 우회하거나 평가용 Torch2.8.0+cu128 lock/프로토콜을 바꾸지는 않는다.
   총 상한$6.80은 유지하며 재개 시 이미 사용한 비용·보존비·회수 여유를 먼저 차감한다.
7. 무료 준비: Accounting30 종료·사후 검증 후 외부 setup/runpod-prep.DK1IeD의
   검증된 19개 파일을 원본에 통합했다. A/B role·Git SHA·개별 추론 호출 기록·공유 이미지,
   과목별 시간·안전한 bundle/로컬 receipt와 clean-clone 재현 명령이 포함된다.
   CPU40/lint/typecheck PASS. bundle·LOCAL_VERIFIED receipt·정확한 삭제 대상 gate,
   staged-byte secret 검사까지 검증했다. clean-clone 재현 스크립트·README를 작성했고
   중단된 stage의 로그 충돌 방지·실제 단일4090/VRAM 확인·팀 origin 제한을 구현/CPU 검증했다.
   명령: 준비본에서 `PYTHONPATH=src MMDL_PYTHON=<검증 venv>/bin/python bash scripts/test_reference.sh`.
   README는 clone shell과 무변경 dry-run을 구별하고 기존 checkout의 재개 명령을 명시한다.
   검증 venv를 bootstrap Python으로 지정한 `reproduce.sh` shell 진입점 dry-run도 exit0 확인.
   기존 SMOKE 열람본을 Windows Edge headless로 실제 렌더링했다(외부 setup/smoke-one-viewer.png).
   토큰 ID가 질문/응답보다 먼저 표시되는 사용성 문제를 준비본 render에서 보완했다.
   합성 2건의 질문·A/B 보기·공유 PNG·응답·null 설명·접힌 metadata를 Edge에서 확인했다.
   증거: 외부 setup/viewer-preflight.{png,json}, setup/runpod-prep-checks.log.
   원본 샘플/HTML은 수정하지 않았고 실제 A/B viewer 검증은 아직 아니다.
   통합 원본에서도 CPU40/Ruff/mypy/bash 검사 PASS. 환경·backend·parser·P0·generation 값은 그대로다.
   `smoke-structural` PARTIAL 3/3 완료, exit0/시스템 실패0. 사전 선택한 각 ID를 한 번씩 실행했다.
   Architecture_and_Engineering_14: open/1 image, 2,048 tokens/length/1,183.360초.
   Biology_29: 5 images, 457 tokens/EOS/269.916초.
   Music_21: 2 images/처리 pixel 합 2,591,744, 346 tokens/EOS/273.765초.
   전체 1,752.717초. 입력 tensor CPU 재구성/hash, P0/chat, 실제 token decode, PNG 8개,
   이미지 순서, raw 재채점, 과목별 산술/시간, JSONL/public hashes, inline HTML, 3개 독립 호출 기록 PASS.
   통합된 source 31개 hash 일치. 증거: 외부 `setup/smoke-structural-audit.json`, `setup/smoke-structural.log`.
   peak RAM RSS 11,503,804,416 bytes. CUDA allocator allocated 9,001,974,784 / reserved 9,347,006,464 bytes.
   이 allocator peak는 WSL이 보고한 물리 VRAM 8,546,484,224 bytes보다 크다. OOM은 없었으나
   실제 물리 residency/host paging은 계측하지 않았으며, allocator 수치를 물리 VRAM 상한으로 해석하지 않는다.

## 결정

- sc-load/메모리 조회에 기존 작업 기록 없음. 현재 파일을 기준으로 시작.
- 자료 조사와 장비 확인은 병렬, 설치→probe→GPU 실행은 순차 수행.
- 코드와 작은 결과는 저장소, 환경·모델·벤치마크·대형 결과는 WSL Linux 저장.
- P0 한 종류만 사용. sampling 및 benchmark 조건은 점수에 따라 변경하지 않는다.
- Ponytail: 기존 표준 라이브러리와 Transformers/Datasets를 사용하며 웹앱은 만들지 않는다.
- 진행 상태는 이 문서 하나에 기록. 정답/benchmark 내용을 대화로 출력하지 않는다.
- 새 요청이 이전 클라우드/게시 금지를 대체한다. 단일 RunPod 용도·총 USD 6.80 예산과
  팀 URL·SSH 인증은 확인했다. 실제 단가/최대시간·SSH 회수·watchdog/삭제 gate 전에는 배포하지 않는다.
  RunPod에는 로컬 코드/환경/모델/cache를 업로드하지 않고 GitHub 고정 commit만 취득한다.
- RunPod 공식 manage-pods/storage/types/pricing 문서를 확인했다. Stop은 볼륨을
  보존하지만 storage 과금이 남으며, 검증 완료 후 정확한 작업 Pod를 Delete해야 한다.
  Network Volume은 기본 생성하지 않는다. MCP/v2에 예약 stop 옵션은 확인되지 않았다.
  SSH와 독립된 외부 MCP watchdog의 시간 한도/실제 stop 증거를 배포 전에 준비한다.
  외부 실행 셀의 5초 타이머 모의 검사는 완료했으나 실제 Pod stop 검증은 아니다.
  4090 조회 가격은 community $0.34/h·secure $0.74/h, stock LOW였으며 확정 견적은 아니다.
  공식 `runpod-torch-v280` template과 Docker registry metadata 확인 완료:
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Linux amd64 digest
  `sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`.
  해당 image build history의 Python3.12/venv 설치를 확인했으며 실제 Pod 실행은 미검증이다.
  사용자가 이번 작업 예산의 사용 가능 여부를 확인했다.
  설치·평가·회수·storage·복구 여유를 합친 총 상한은 USD 6.80이다.
  잔액 소진 전에 GPU를 중지할 여유를 확보하며 자동 충전/결제 설정은 변경하지 않는다.
  MCP 사용내역 조회에는 계정 잔액 필드가 없으므로 이를 잔액 조회 증거로 사용하지 않는다.
- 샘플 JSON과 자체 hash를 한 번의 원자적 저장으로 함께 보존한다.
  최초 smoke에서 이미 실행 중이던 이전 sidecar 형식은 읽기만 호환한다.
- CUDA peak는 해당 프로세스 시작 후 누적 최대치이며 샘플별 독립 peak가 아니다.
- 첫 smoke 이후 실행에는 benchmark 본문 없이 생성 token 수/경과시간만 로그에 표시한다.
- 추가 smoke는 정답 무관 구조 기준으로 Architecture_and_Engineering_14(open),
  Biology_29(5 images), Music_21(처리 pixel 합 최대)을 선택한다.
- 속도 점검(읽기 전용): Accelerate 1.11.0 `dispatch_model`/`AlignDevicesHook`는
  CPU decoder weight를 각 forward 전에 GPU로 보내고 이후 meta로 돌린다.
  20개 decoder weight 합 약 3.760 GiB/token은 safetensors 크기로 계산한 전송량이며
  실제 PCIe 계측값은 아니다. 표준 pinned/non-blocking inference switch는 없으므로
  현재 실행을 유지한다. 추가 device-map 튜닝·A/B 실험은 실행하지 않았다.

## 다음 행동

1. Accounting30 종료·사후 검사 완료. 원본 결과 보존, 재추론하지 않음.
2. 격리 준비본 통합·CPU 검사·주관식/다중 이미지 3개 검증 완료. 추가 로컬 추론 없음.
   build_messages가 run_dir/prompts 사본을 쓰는 보완도 포함하며 P0는 바꾸지 않는다.
3. 3090 명시적 프로필/장비 검사 CPU 검증·새 고정 commit 게시. 이전 5090은 호스트 CUDA 실패로 정리 완료.
   기존 실행 commit의 깨끗한 별도 local worktree를 보관했다. 최종 bundle 검증에는 실제 실행 SHA의 worktree를 쓴다.
4. 사용자의 재개 지시에 따라 4090/3090을 사용한다. 대여 직후 libcuda cuInit/UVM 열기부터 확인하여 설치 낭비를 막는다.
   삭제된 Pod는 --resume하지 않는다. 정상 새 Pod/남은 예산/독립 watchdog을 확인한 뒤 README의 clone 명령을
   MMDL_CODE_COMMIT에는 새 게시 전체 SHA, MMDL_JOB_ID=mmdl-val-20260922-3090,
   --hardware configs/hardware/rtx3090_24gb.yaml을 지정한다(4090 배정 시 대응 profile 사용).
   doctor·최소 smoke 완료. 현재 A900 진행 중이며 B900·검증/포장/회수/삭제가 남았다.
전체 protocol은 FROZEN이다. 상태는 RUNPOD_ASSIGNMENT_RUNNING.
A/B·전체 결과 로컬 회수·실측 보고서·최종 CLEANUP_VERIFIED는 아직 달성하지 않았다.
