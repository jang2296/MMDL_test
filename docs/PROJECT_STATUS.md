# Assignment 1 baseline — 작업 상태

## 2026-10-18 20:14 KST — import 수정794 CPU/새 입력 통과·동일Pod 재검증 준비

Linux 코드복사본68 SHA를 확인하고 전체794검사296.668초/1환경skip과 입력24검사42.116초를
통과했다. Windows마운트 전체검사는 I/O 지연으로 소유process만 중단했으며 완료로 세지 않는다.
19집중검사·최종test명 cold회귀1개4.168초/Ruff/bounded mypy1/게시안전3도 PASS다.
full의 test본문과 최종본은 identifier-only 차이이며 hf_문자열의 게시secret검출 오탐을 피했다.
외부의존성-wide mypy의 기존psutil stub부재1건은 별도 한계다. 새prepared_v3 검증117.343초는
16적격/차단0·동일10distinct/단일5복수5이며 원래76입력/10source/선정/probe/protocol은 불변이다.
독립 코드·입력delta 검토 PASS이며, 기존 승인된 로컬전체/원격변경부 검증 방식으로 새 clean SHA/
같은venv의 정상lock/pipcheck·변경부·CUDA/BF16·13모델 재해시 후에만 재실행한다.
유일한production변경은 shared_smoke.py의 신뢰dependency 초기화 순서다. 실제GPU update0/
reload미검증·원래금융기한은 유지한다. 교수님MMDL/V8·과거실패원문은 불변이다.

## 2026-10-18 20:05 KST — shared HF cold import guard 충돌·수정 검증 중

첫 actual shared capture는247.452초/exit2이며 attempted0/update0/checkpoint0이다.
build_policy의 첫AutoProcessor import가 Transformers의 optional openai find_spec을 실행해
기존teacher guard에 차단됐다. 직접API/teacher 호출이 아니며 load_base 이전이다.
실패launch/result/log3파일의 원격/로컬SHA를 검증해 별도 보존했다. 실측peak33,554,432bytes는
runtime BF16 probe값이므로4B 모델 적합성으로 해석하지 않는다.

최소수정은 정확runtime검사 후/guard설치 전 HF·PEFT4심볼의 신뢰startup import다.
guards.py·프로토콜·모델·데이터·정밀도/이미지/출력·보상·모듈범위는 불변이고 실제모델/processor
생성과 actor/vault 작업은 guard 뒤다. cold subprocess의 실제tiny정책과 teacher3진입·vault·
benchmark차단을 포함19집중검사8.520초 및 독립검토 PASS다. test-only 첫 초안의stateless GELU
오인은 기록하고 파라미터 등록 여부로 구분했다. 전체CPU/새code결합 입력 승인과 정상Git게시·
clean clone/환경검증 뒤에만 재실행한다. 원래금융/GPU기한은 늘리지 않으며 새학습 성공은 아니다.
증거 private retry-v1/runtime/failed_capture_local_verification_v1.json 및 hf_import_fix_* 로그다.

## 2026-10-18 19:56 KST — 새3090 정상setup·793+24검사 통과, 제한실행 제출

사용자 재시도로 생성한 소유Pod `2uiqg6ym30ou3y`는 이전 UVM장애와 다른 실제3090
UUID0481210a…/driver570.211.01이며 CUDA/BF16을 통과했다. MMDL_test clean8e9a8ee…
직접clone과414개 비공개 입력/2,065,615,841bytes·68code SHA를 검증했다. 지정13모델을
병렬 취득·검증한 뒤 정상setup도 재검증했다. 같은lock14개 NVIDIA wheel의 공식SHA
선설치 후 변경 없는setup491은2786.392초/exit0,793+24 CPU검사/pipcheck/doctor/모델
검증으로 닫혔고4개 로그·환경 파일의 원격/로컬SHA가 일치한다. doctor RAM은 host값이며
실제cgroup RAM29,999,996,928bytes와 구분한다.

10:55:45UTC 기존 shared_smoke --execute→별도process --reload-check를 예산watchdog3234/
capture3236으로 제출했다. 고정10distinct/단일5·복수5/ALG-NB3/최대2update이며 새900이나
대규모학습이 아니다. 두 production 입력검증과 runtime/GPU/모델 검사를 생략하지 않는다.
현재는 실행 전 입력검증이며 실제4B update/reload 성공은 아직 아니다. 최초과금09:59:13.359UTC,
GPU11:59:13.359/job11:57:13.359/recovery12:14:13.359UTC·0.30USDh/reserve0.20은 불변이다.
10:51:04UTC 실제잔액0.7152243244USD·현재과금0.241USDh다. 근거는 private retry-v1/runtime의
remote_input_sha_verification_v1.json 및 setup_local_verification_v1.json이다. 교수님MMDL/V8는 불변이다.

## 2026-10-18 18:42 KST — 3090 최초 CUDA/UVM 사전검사 실패·자원 삭제

실험 코드는 MMDL_test `ce2bc35edaf80f539641d4ae64c7fa7d1b0b1f4c`에 정상 게시했다.
명시30경로 byte/mode·동봉 manifest/lock/P0와 게시 검사, checkout64검사/34.602초,
EOF-only delta14검사/30.617초·Ruff/mypy6 및 독립 검토를 통과했다. 과거 전체570 CPU
증거는 production172항목 중171동일/나머지 추가EOF newline 제거 전이며 이를 새 전체
재실행으로 부풀리지 않는다. 현재68code 결합 입력 승인112.168초/차단0·10distinct
structure(단일5/복수5)·414개 비공개 전송목록은 로컬 준비/검증일 뿐 원격 실행이 아니다.

09:37:00.141UTC 소유Pod kxxt6w3fpsb7ur를0.22USDh의 RTX3090으로 생성했다.
실제3090 24576MiB/driver580.178.04·libcuda580.178.04, 공식 컨테이너CUDA12.8.1을
확인했으나 Torch뿐 아니라 직접cuInit도999였다. nvidia1/ctl open은 정상인 반면
/dev/nvidia-uvm open은 errno5 EIO이며 환경변수 제거 후에도999였다. 이 시점에는
코드clone/입력전송/설치/모델다운로드/추론/update/checkpoint가 모두0이다. 원인은
호스트/컨테이너 UVM 장치 접근 단계로 좁혀졌으나 정확한 kernel 원인은 미확정이다.
VRAM 적합성/실제4B RL/reload는 검증하지 못했다. 공유 driver/reset/재설치/조건축소나
변경 없는 동일 실패 재시도·추가 유료Pod는 진행하지 않았다.

두 진단로그1997/212bytes의 SHA와 실identity·boot41줄을 로컬 보존·검증한 뒤 소유Pod만
delete204→get404로 삭제했다. cluster포함 Pod0/NetworkVolume0/전체페이지 완료를 확인했다.
container30GB와Podvolume120GB는 폐기했지만 입력/모델/실험결과는 원격에 없었고 로컬 준비본은
모두 보존된다. 09:42:09UTC GraphQL 잔액0.9107895021USD/currentSpend0.241USDh는 삭제직후
지연snapshot이며 최종청구 또는0과금 인증이 아니다. 이전0.9286258243USD와 구분한다.
상태는 HARDWARE_PREFLIGHT_FAILED/CLEANUP_VERIFIED이고 실험 완료/학습 성공이 아니다.
09:43:04UTC 후속 GraphQL은 같은잔액0.9107895021USD·현재과금0USDh로 확인됐다.

근거: `/home/jang2/mmdl-artifacts/notebook-rl-3090-20261018/runtime/`의
`cuda-diagnostic-v1.log`, `cuda-device-access-v1.log`, `predelete_local_verification_v1.json`,
`owned_pod_cleanup_final_v1.json`. 새 code 실행SHA는 ce2bc35e이며 후속 상태문서 commit과
구분한다. 교수님MMDL/V8 558/900·기존 원문/가중치는 불변이다.

## 2026-10-18 18:20 KST — 새3090 제한 시험 승인·배포 전 로컬 준비

사용자가3090 RunPod로 시험하며 계속 추적하도록 승인했다. 대상은 공유HF10문항/최대2update와
독립프로세스reload이며 대규모학습/900평가는 아니다.09:15:17UTC 잔액0.9286258243USD·현재
과금0·계정Pod0/NetworkVolume0을 실제 조회했다. GPU catalog Community3090은0.22USDh이며
재고는NONE/일시LOW로 변동했다. 마지막확인 재고를 다시 읽은 뒤 생성한다. 저장비 포함운영상한
0.30USDh/회수예약0.20USD/최초과금부터GPU최대2시간+회수15분을 고정한다. 남은잔액으로
성공/완주를 보장하지 않으며 자동충전·별도NetworkVolume·다른GPU를 생성하지 않는다.

기존private train16의76입력/10measured source·제외 검사를 유지하고 새68codehash를 결합한다.
기존single/two probe가 같은structure를 공유했으므로 별도의eligible distinct probe와10distinct
structure를 동결한다. 새로운CPU입력승인·게시SHA·실제device/재정watchdog 결합 전 GPU시험은 없다.
교수님MMDL/V8는 불변이다. 실험용MMDL_test에는 검증된fine_tuning/코드·테스트와 상태문서만
정상게시하며 관련 없는 팀원문서/benchmark파생물/원문/weights는 복사하지 않는다.
재고/실행상태·진행·오류·예산을10분마다 조회하고 회수·로컬검증 후 전용Pod만 삭제한다.
원본증거는 `/home/jang2/mmdl-artifacts/notebook-rl-3090-20261018/`에 별도 보존한다.

## 2026-10-18 16:10 KST — 24GB용 단일 공유모델 HF 경로 구현·검증

`fine_tuning/notebook_rl/shared_smoke.py`는 새 `hf_shared_policy_24gb_v1` 실행 profile이다.
하나의 live BF16 PEFT 정책이 순차 생성/기존 RL 업데이트/adapter-disabled F0 reference를
공유한다. 두 번째 vLLM 가중치와 고정10GiB KV 예약 및 merged weight push를 제거한다.
기존 native 경로와 `protocol_v1.yaml` SHA653fd442…·train.py/modules.py는 byte 불변이다.
모델/revision·이미지1003520..4014080·context16384·solution3072·T1/T2/T3 수·보상·파라미터
분할/optimizer는 자동 변경하지 않는다. 별도최대10문항/2update 연결 시험이며 일반 학습이나
기존 vLLM G0–G13 통과를 의미하지 않는다. ALG-NB3와 ALG-NB+D는 원래 turn/DIAG 차이를
유지한다. 사후 감사용 노트의 추가 호출은 새 실행 binding에 명시하며 loss/라우팅 근거가 아니다.

원문 group/tree는 write-once 원장으로 남고 신호 없는 폐기 tree에도 최종 노트 상태를 보존한다.
실제 회복/advantage가 없으면 이를 발명하지 않는다. initial/post-update one/two-image logprob
probe를 강제하고, 실패 sync/불완전 cohort/예산 중단/0update를 완료로 판정하지 않는다.
별도 reload 명령은 capture PID 종료, 현재 code/input/profile binding, 전체 parameter hash,
optimizer step 및 동결된10 ID/cursor를 검사한다. 메모리 계산10.42GiB는 상주 하한/보수적
trainable allowance일 뿐 동적KV·activation·workspace·생성 점수 tensor를 포함한 peak가 아니다.

검증: 정상 PYTHONPATH 전체570 CPU tests/253.714초 exit0·Ruff·범위 mypy5·독립 Sol 검토 PASS.
첫 전체검사570 중 기존 checkpoint subprocess1개는 잘못된 명령의 PYTHONPATH 누락으로
실패했다. 소스 변경 없이 README 환경으로 해당1개/7.017초와 전체570을 다시 통과했고 두 로그를
보존했다. checkpoint cursor의 필수 epoch/discarded IDs 누락은 개발 검토에서 수정 후 실제
tiny Adam/checkpoint.save/load_resume 검사로 확인했다. 새 launcher의 GPU/rollout 조립 시험에
쓰인 제어값은 authored fixture이며 실제4B 실행 증거가 아니다.

별도 로컬5060 CUDA 시험은 tiny random Qwen BF16 기저/FP32 master의 생성4개·직접기울기
업데이트1개·전후 두 probe delta0·frozen base 불변을 실제 확인했다. peak allocated68,595,712 /
reserved69,206,016bytes는 이 tiny 시험 값이며 4B로 외삽하지 않는다. 학습 checkpoint를 저장하지
않았고 프로세스 종료 후 로컬GPU0MiB/0%를 확인했다. 기존 실제4B RL update0 상태는 유지한다.

증거 정본 `/home/jang2/mmdl-artifacts/notebook-rl-shared24-20261018/`:
`sources_v1.sha256`172항목 SHAecf43a8a… 및 전후 재검사, `full_cpu_v2.log` SHA c05abfd9…,
`full_cpu_v1.log`, `checkpoint_env_recheck_v1.log`, `tiny_cuda_probe_v1.json`,
`independent_review_v1.json`, `ruff_v1.log`, `mypy_v1.log`, `verification_v1.json`.
데이터 전수검사90개는 이전 변경 없는 suite의 기록이며 이번에 재실행했다고 쓰지 않는다.
새 파일로 code hash가 바뀌었으므로 이전 데이터 승인/Pod job을 재사용하지 않는다.

실제 pretrained4B의3090/4090 메모리/속도·10문항 RL·새 프로세스 GPU reload는 미실행이다.
후속은 코드 게시/clean SHA→새10개 및 source 결합 승인→실제 장비/예산 검증→제한 smoke다.
새 유료 Pod/모델 다운로드/게시/새4B checkpoint/가중치 삭제는0이며 잔액 약0.93USD로 완료를
보장하지 않는다. 교수님 MMDL clean197838c1·MMDL_test clean8595d2ce·V8 558/900은 불변이다.

## 2026-10-18 15:35 KST — 실제 Qwen4B·약10문항 시험은 로컬 GPU 용량에서 차단

사용자가 실제 checkpoint의 문제풀이·정오 확인·재시도·사후 오답노트·검증 기반 V/M/L 또는
M/L RL 업데이트를 시험하고, 성공 후 이번에 만든 변형 가중치만 제거하도록 요청했다.
현재 pinned 원본13파일의 크기/SHA를 실제 캐시에서 전수 검증했으며 재다운로드는 없다.
BF16 serialized weights8,875,719,344bytes가 로컬5060 전체VRAM8,546,484,224bytes를
초과한다. `launch.py`의 두 가중치+고정10GiB KV만28,488,856,928bytes/26.532GiB이고,
이것은 activation·adapter·gradient·optimizer를 제외한 하한이다. 실제CUDA BF16 작은 연산은
통과했으므로 CUDA 설치 장애와 구분한다. 모델학습 할당/실제 응답/optimizer update는 모두0이다.

현재 native launcher는 CUDA/owned financial watchdog/clean SHA/기존 데이터 gate를
요구한다. 이를 끄거나 양자화·해상도/출력/KV 축소·device_map 학습 offload로 우회하지 않았다.
CPU가 본질적으로 불가능하다는 결론은 아니다. 기존 별도HF sampler+policy+전체 export는
RAM도 크게 쓰며, 공유PEFT CPU sampler·BF16 CPU/FP32 master 산술·동기화·실제 peak/runtime를
별도 구현·검증해야 한다. tiny CPU 시험의 FP32 기저 결과를 실제4B BF16 CPU 성공으로 쓰지 않는다.

같은 native 학습을 검증하려면 충분한VRAM의 GPU와 새 유료 실행의 비용 상한·최대시간 확인이
필요하다. 현재는 새 cloud 실행/게시/학습 성공/새 checkpoint/가중치 삭제가 없다. 승인 후에도
기존 program40/train16의 승인 범위를 새 protocol/source에 자동 재사용하지 않고 새10개 입력과
노트 생성·DIAG 학습 설정을 결합해야 한다. 모든 답이 같아 advantage가 없거나 실제 회복이 없으면
그 상태를 보고하며 가짜 오답·회복을 만들지 않는다. 성공·재로딩·증거 확인 뒤 새 시험weights만
삭제하며 원본 HF 캐시·기존 checkpoint·원문 기록은 보존한다.

근거 정본: `/home/jang2/mmdl-artifacts/notebook-rl-real10-20261018/local_preflight_v1.json`.
MMDL clean197838c1/MMDL_test clean8595d2ce와 V8 558/900은 불변이다.

## 2026-10-18 02:23 KST — 로컬 오답노트·라우팅 RL·폴더 단독 실행 검증

이번 목표인 **로컬 소규모 소프트웨어 연결 검증**과 이전 미완료인 native4B GPU 학습을
구분한다. `fine_tuning/notebook_rl/rollout.py`의 선택적 최종 노트는 회복 경로의 원문 token을
보존하고 예산/문맥/빈답/잘림/미회복 상태를 기록한다. `batches.py`는 예산 종료를 전파하고
advantage가 없어 버린 문제도 원문·노트를 보존한다. 감사 노트는 loss/라우팅과 분리하며,
class-balanced DIAG 행동의 프로그램 검증 보상은 실제 RL update에 별도로 연결한다.
기존 frozen protocol을 자동 변경하지 않았다. 노트 활성화는 새 protocol 및 호출 예산이 필요하다.

`test_notebook_rl_local_pipeline.py`에서 실제 작은 Qwen VLM의 이미지 forward/logprob와
생산 RL 손실로3회 가중치를 갱신했다. 첫 Vision ON은V/M/L, 다음OFF는M/L만 변화하고
이전Adam momentum에도V hash는 유지됐다. 두 원인 class의 DIAG 보상·비균일 advantage를
세 번째 update에 포함했고 SFT/CE 진입점을 금지했다. 고정 reference/기저는 불변이다.
별도 Python 프로세스의 checkpoint reload는 모든 parameter hash·step3·cursor가 일치했다.
성공 경로 문장은 **의도적으로 작성한 제어흐름 fixture**이며 실제4B 자기수정 결과가 아니다.
별도 실제 무작위 tiny 생성6개는 실패로 남겨 거짓 회복을 만들지 않았다. 직접 optimizer
검사의 상태는 `UPDATED_UNSYNCHRONIZED`이며 native vLLM 증거로 승격하지 않는다.

`resources.py`·`input_preparation.py`와 동봉한 byte동일 manifest/lock/P0를 이용하여
fine_tuning만 복사한 환경에서 src/env/manifests/prompts 없이 notebook-RL 연결 시험을
통과했다. 외부 `mmdl`/`experiments` import도 차단한 검사다. 모든 과거 fine_tuning 명령의
독립성이나 의존성 설치 없이 실행 가능함을 주장하지 않는다. 이전 train.py/modules.py와
protocol_v1.yaml은 각각 기존 SHA를 유지한다.

- 고정 소스:167파일 전수 SHA 일치.
- notebook 전체 CPU:559 tests/315.178초/exit0.
- 별도 데이터 환경의 생성기·adapter:90 tests/627.343초/exit0.
- Ruff 및 Sol 독립 완료 감사:PASS. 완료한 소스/증거만 판정했다.
- 변경부8파일 mypy는 기존 `scripts/test_reference.sh`와 같은
  `--follow-imports=skip --ignore-missing-imports` 정책으로 PASS다. 별도 엄격 호출은
  외부 qwen_vl_utils의 type stub/py.typed 부재1건을 보고했다. 외부 라이브러리 및 untyped
  함수 본문까지 전부 정적 검증했다고 주장하지 않으며 runtime guard/test는 변경하지 않았다.
- 이전 두 전체검사 실패는 child의 bare module명 때문이며 실패 원문은 보존했다.
  canonical reload-worker 경로 수정 후 직접 실행과 unittest discover 모두 통과했다.

로컬 증거 루트는 `/home/jang2/mmdl-artifacts/notebook-rl-local-20261018/`이다.
`contract-final-v4/verification.json` SHA는
`2aeec54206395bc7f37ba6e9462e159c9d54ff72b47c51971d9c02dd63143d88`,
`full_cpu_final_v3.log` SHA는
`8efc649d8748836da2ebbceb18995ffa0bed0f1e8e773888f389dc791b8136e8`,
`data_cpu_v1.log` SHA는
`116a90f9faf1d4649d0da0e06523277d801d097893a30547a4837e4b38d9effb`다.
요구사항별 완료 감사는 같은 루트의 `local_completion_verification_v1.json`에 있다.
재현 명령과 해석 범위는 `fine_tuning/notebook_rl/README.md`의 local 검증 절을 따른다.

다분야 본학습 pool은 아직 승인되지 않았다. 기존 program graph40(train16/dev8/sealed8/
style_sealed8)의 제한 AI 검토·제외 검사는 그 범위에만 유효하다. 새 source/protocol/input에
예전 승인을 재사용하지 않으며 TabMWP/Geometry3K/PGPS9K 및 실제 ChEBI 원천의 남은
출처·권리·제외·검토 조건을 완료했다고 표시하지 않는다. 데이터 생성기 시험은 학습 승인이 아니다.
이번 RunPod·모델취득·GitHub게시0, MMDL clean197838c1/MMDL_test clean8595d2ce 및
V8 558/900은 불변이다. 실제4B RL/GPU동기화·reload/PhaseA/150·성능개선은 별도 미완료다.

## 2026-10-17 15:35 KST — 소유 Pod·작업 볼륨 삭제, 실제 학습 검증은 미완료

독립검토에서 두 실패 실행14파일·환경8파일·job2개 및 복구400파일의 실제 로컬 SHA와
기존 원격 receipt 일치를 확인했다. 소유Pod e10034q5chqqcr를 삭제204→get404,
계정Pod0(cluster 포함)/NetworkVolume0/추가페이지없음으로 확인했다. container30GB와
workspace120GB의 설치환경·모델cache는 원격 복구 불가이고 로컬 입력·코드·실패 기록은
보존됐다.06:34:37UTC 공식 GraphQL 잔액0.9286258243USD/현재과금0USDh를 직접 확인했다.
결제·자동충전은 변경하지 않았으며 최종 청구서는 아니다. 증거는 private runtime/
owned_pod_cleanup_final_v1.json과 goal_blocked_audit_v1.json이다.

코드 수정4185db7·문서포함 clean 게시8595d2ce의 CPU검증은 완료했지만 실제 GPU 재검증,
RL update/reload/150평가는 미완료다. update0이며 학습 성공을 주장하지 않는다.
16:10 연장은 승인되지 않았고 새 GPU 실행·종료시각 승인이 필요하다. 동일 권한 차단이
3연속 goal turn에서 반복됐고 GPU 검증 전 가능한 독립 준비·보존 작업은 완료했다.
교수님MMDL clean HEAD197838c1·기존 V8 558/900은 불변이다. 이 기록은 내부 상태이며
게시8595d2ce 이후 운영기록만 추가했고 실험 코드나 교수님 저장소를 변경하지 않았다.

## 2026-10-17 15:30 KST — 중지 후 재개 입력 복구 원장 검증

추가 GPU 실행 없이 로컬 원본385개/2,064,900,296bytes를 전수 stream SHA 검증했다.
누락·변경·위험 경로0이다. 기존 전송387개와 최신 v4 승인13개를 합친400개,
2,065,510,375bytes 및 새 metadata2개를 포함한402개 명시적 전송 목록을 고정했다.
private runtime/private-restore-inputs-v1.json SHA는 c2b553c6ec58e2d4a8cb2ce2f39cc40a345e59aeca6b8df0371d0fa956a5bfe3,
files-from SHA는30acafef67423965940f2d498697a130b763f839d82e51c50fca18e44ea7cbaa다.
생성기 assert/roundtrip/Ruff와 독립 연결 검토 PASS이며 기존 원본·원장은 불변이다.
code/venv/baseweights/credential은 전송 목록에 없고64개 source binding은 clean GitHub
8595d2cebb01cdc93c4d0463a70ab3354c05df88와 일치한다. 이는 로컬 복구 준비 증거이지
실제 원격 전송·destination safety·GPU 학습·새 실행 승인이 아니다. 실행 재개에는 새 시간
승인이 필요하며 현재15:40 한도를 유지한다. 실제 RL update0/reload 미완료는 그대로다.

## 2026-10-17 15:24 KST — 로컬 수정 검증·게시 완료, GPU 중지·연장 승인 대기

메모리 시작 비율0.7 최소 수정은 로컬771 tests(1skip,325.963초), 입력24 tests(23.344초),
집중19/Ruff/mypy1 및 독립검토를 통과했다. MMDL_test4185db75ec502b50990a6c09734dfe96b0772153
정상 push/원격HEAD·clean 확인. v4 실제 입력 preflight는16개/차단0/130.005927초 PASS다.
수정 후 실제 GPU 실행은 아직 없으며 RL update0/reload 미완료/150 보류다.
기존 native G9/G10 통과 및 vLLM 시작 실패7파일979514bytes를 로컬 SHA 검증했다.
유휴 GPU 과금을 줄이기 위해 소유Pod e10034q5chqqcr를06:22:31UTC 중지, EXITED를 재확인했다.
120GB /workspace는 유지되나 저장비는 계속 발생한다. 컨테이너 임시 code/입력 복사본은
중지 시 폐기되므로 재개할 때 GitHub clean clone·private 입력 재전송·실제환경 재검증이 필요하다.
사용자의 잔액 질문은 시간 연장 승인이 아니다. 기존15:40·총2.30 한도는 유지한다.
MCP REST v2 밖의 공식 GraphQL로 실제 clientBalance/currentSpendPerHr를 조회할 수 있다.
기존 API key는 메모리에서만 사용하며 출력/전송/저장/결제설정 변경하지 않는다.
실제 잔액·상태·중지/학습 미완료 증거는 private runtime/에 보관한다. 교수님MMDL/V8 불변이다.

## 2026-10-17 15:17 KST — 기울기 문제 통과 후 vLLM 시작 조건에서 중단

76cfc4e의 strict deterministic native G9/G10은 실제 통과했다. 다음 vLLM 생성기가
기본 gpu_memory_utilization0.9로 가용38.06GiB보다 큰42.66GiB를 요구해 시작을 거부했다.
HF 학습 모델을 같은 GPU에 올린 구성을 고려하지 않은 시작 설정 누락이며 실제 OOM이 아니다.
vLLM0.11 공식 gpu_worker.py는 시작 조건 후 명시KV를 그대로 사용한다. 시작 비율0.7만
명시하고10GiB KV/16동시/16K문맥/BF16/이미지/출력은 유지하는 최소 수정을 준비 중이다.
19변경부 CPU검사/Ruff/mypy1·독립검토 PASS, 로컬 전체 CPU검사는 진행 중이다.
원문은 private runtime/failed-native-vllm-startup-v2/에 회수 중이다. 새 update0이며
학습 checkpoint/reload/150 성공이 아니다. 연장 질문 전 기존총2.30/15:40/reserve0.50을
그대로 적용한다. 교수님MMDL/V8는 불변이다.

## 2026-10-17 14:53 KST — native G9 실패 보존·검증 절차 재사용 승인

실제 A40 G1과 G3 초기 차이0 이후 G9에서 OFF/직접동결 기울기가 불일치하여
05:41UTC exit2로 종료했다. optimizer update0, 학습 checkpoint/reload/150은 미완료다.
실패 원문은 private runtime/failed-native-g9-v1/에 회수했으며 수치가 기록되지 않아
CUDA backward 비결정성을 확정하지 않는다. strict deterministic 연산과 OFF→OFF 대조,
오차/기울기 norm 진단을 추가했고 기존 rtol1e-6/atol0·BF16·데이터 gate는 유지한다.
수정32 CPU tests/Ruff/mypy3·독립 검토 PASS, 로컬 전체 CPU검사는 진행 중이다.
사용자는 로컬 전체검사+원격 변경부 집중검사·정확 lock/pip/CUDA BF16/모델13파일
재검증을 승인했다. 새 GitHub clean SHA만 실행하고 이전 원격765+24 증거는 보존한다.
총2.30USD/최초과금/회수예약0.50USD/GPU06:40UTC/job06:38UTC는 바꾸지 않는다.
새 RL 성공/모델 개선/전체 MMMU 성적으로 보고하지 않으며 교수님MMDL/V8는 불변이다.

## 2026-10-17 14:28 KST — 정상 setup 완료·실제 native controller 제출

75c46dc의 정상 setup이962.941995초/exit0으로 종료했다.765 CPU tests840.431초와
입력진단24 tests5.610초가 모두 PASS이며 실제 A40 CUDA/BF16 probe도 통과했다.
지정 모델13파일 SHA·revision ebb281ec70b05090aa6165b016eac8ec08e71b17을 확인했다.
닫힌 doctor/model/pip-freeze/setup log4파일을 회수하고 원격/로컬 SHA를 모두 대조했다.
실제 CUDA 가용메모리 단위의 total은50,897,289,216bytes로 읽었고 nvidia-smi 표시와 구분한다.

실제 Pod/새 clean SHA/모델 receipt/lock/device UUID를 결합한 job SHA는
c9c73cef8913e789ed98693ab2a2c310091c7346b696be365ecf768e3c08ee22다.
05:28:12UTC native watchdog2919/coordinator2920을 시작했다. 이는 입력 preflight 단계의
실제 제출이며 아직 GPU 학습·가중치 갱신·reload 성공이 아니다. 새로운 사용자 승인에 따라
compute06:40/job06:38UTC/reserve0.50USD를 적용했고 최초과금/2.30USD는 그대로다.
근거 private runtime/setup-75c46dc-local-verification-v1.json, native-launch-v1.json,
setup-75c46dc-closed-v1/, native-job-v1.json이다. 교수님MMDL/V8은 불변이다.

## 2026-10-17 — 사용자 명시 승인: 동일2.30USD 안에서15:40 KST까지 연장

사용자가 `동일 2.30 USD 안에서 15:40까지 연장`을 답했다. 새 native 실행에 적용할
compute cutoff는06:40:00UTC, 내부 job deadline은06:38:00UTC, 회수예약은0.50USD다.
최초과금04:01:36.102UTC/총2.30USD/운영상한0.65USDh/소유Pod e10034q5chqqcr는 불변이다.
기존 setup1934의 watchdog/실행 코드는 hot patch하지 않는다. 정상 setup 완료 뒤 새 native
watchdog와 private job에만 새 승인을 결합한다. 모델·sampling·데이터·검증 기준/최대2update는
바꾸지 않으며 추가Pod나 미검증150평가를 자동 시작하지 않는다. 실제 RL/reload는 아직 미실행이다.
승인 원장은 private runtime/user-budget-extension-v1.json에 보존한다. 이 절이 이전 시간·예약
기록보다 우선하며, 교수님MMDL/V8는 불변이다.

14:24KST 사용자는 현재 잔액 약1.50USD를 보고했다. 이는 추가 예산이 아니라 기존2.30USD의
남은 잔액이며 새 실행 계산에 더하지 않는다.05:24:41UTC 직접 조회에서 Pod RUNNING/GPU0%/
4MiB·CPU unittest PID1971 실제 실행·688개 ok를 확인했다. GPU 학습/모델 준비는 미완료다.
지연 원인은 유료GPU 장비에서 전체 CPU 회귀검사를 반복한 배포 준비 절차이며, 이를 모델 학습
시간이나 GPU 계산 완료로 설명하지 않는다. 기존 실행의 두 test-only 실패와 수정 기록을 보존한다.

## 2026-10-17 14:10 KST — CPU FP32 경계 재현·test-only 수정 게시·정상 setup 재시작

- 두 번째 setup은05:02:07UTC 전체765 CPU tests 중1실패로 종료했다(1424.810초).
  guard10은 통과했으며 실패는 작은 Qwen CPU 시험의 chunk1 projection과 전체 projection을
  bit 동일하게 요구한 비교다. 같은 hidden state에서 CPU GEMV/GEMM 차이로 logprob1개가
  2.384185791015625e-7(1 FP32 ULP) 다른 것을 독립 재현했다. reference/full 비교는 exact다.
- test-only chunk1 비교를 rtol0/atol5e-7로 바꾸고 data-only 시험 worker의 불필요한
  Torch/PEFT import를 분리했다. production G3/G7·보상·가중치·데이터 승인 기준은 불변이다.
  로컬 policy6/guard10 tests·Ruff PASS, 기존33 mypy 진단은 그대로이며 새 진단0이다.
  64개 production source hash와 v2 data scope 및 상위/게시 파일 byte 일치를 확인했다.
- MMDL_test 정상 게시 SHA75c46dc44cfd076fc0b20344d09fc7fa1cea195d와 Pod 직접
  clean clone이 일치한다. 종료한5d99 checkout/실패 로그는 보존했다. 로그 원격/로컬 SHA는
  7264d88fcef25c6360043df83e89b18e14add551aebf49de5ab5c138497da2c4다.
  새 setup1934를05:10:18UTC 시작했다. 기존 venv만 재사용하고 전체 CPU/lock/pip check/
  CUDA BF16/지정 모델13파일 검증은 생략하지 않는다. 현재 실제 RL/reload/150은 미실행이다.
- 원래 소유Pod e10034q5chqqcr·최초과금/2.30USD/reserve1USD/compute06:01:36.102UTC/
  job05:59:36.102UTC는 유지한다. 교수님MMDL/V8은 불변이다. 실제 T2/T3 오답노트와
  nonzero 학습 기여를 관측하기 전 ALG-NB3 label만으로 자기수정 학습 성공이라 하지 않는다.
- 별도 CPU 입력 원장 `evaluation150-v1/`은 실제 V8 하위5과목150 ID/168이미지의
  P0·8K·문항 seed/원본 SHA·이미지 역할·gold-blind payload를 고정했다. 이는 GPU평가가 아니다.
  동점은 subject 이름순으로 고정했고 original Materials_15의 해설 전용 이미지는 제외됐다.

## 2026-10-17 13:38 KST — guard 시험 timeout 수정·새 clean setup 실제 재개

- 원격 단일 시험이91.134초/outer90초 TimeoutExpired임을 재현했다. 테스트 actor가
  세 차례 data worker를 실행하므로 outer300/inner60으로 test-only 수정했다.
  전체 guard10검사148.606초/Ruff/범위mypy PASS이며 assertion/production guard는 유지한다.
- 원래 setup205/child560 등은 종료·부재를 확인했다. 닫힌 setup 로그의 로컬/원격
  SHA9a38c6161dd746dd052a59f80951bb09316fb47e67cf48e489a59c7be986c12a 일치다.
  원래 code checkout도 Pod의 `multimodel-code-2b754a8`로 보존했다.
- MMDL_test 정상 push·Pod 직접 clean clone SHA는5d99dd8247ac06e31062f33f51fb66104167bf12.
  새 setup1147를04:37:16UTC 시작했고 기존 canonical venv만 명시적으로 재사용한다.
  전체 CPU/CUDA BF16/모델13파일 검증은 생략하지 않는다. 실제 RL/reload/150은 아직 미실행이다.
- 원격 실제 preflight177.291초의 별도 default120 문제도 제한 smoke dispatcher300으로
  수정·검증했다. 새 D1–D8 v2는115.062초 CPU 검증/0blocker이며 동일 components/원장/
  입력을 보존한다. 새13파일89,275bytes는 원격 전송·모든SHA 검증을 완료했다.
  train16/heldout24·64codehash·probe·budget은 독립 검토했으나 GPU 성공은 아니다.
- 최초과금04:01:36.102UTC/2.30USD/reserve1USD/운영상한0.65USDh/compute06:01:36.102UTC/
  job05:59:36.102UTC는 그대로다. 교수님MMDL/V8는 불변이며 실제 RL/reload 우선이다.
  private 근거는 `notebook-rl-smoke-20261017/runtime/`의 실패·수정 receipts와
  `remote-scope-transfer-verification-v2.json`, `candidates-v1/smoke-data-reviewed-v2/`다.

## 2026-10-17 13:15 KST — A40 대여·고정 SHA clone·비공개 원격 검증, 설치 중

- 소유Pod e10034q5chqqcr/A40는04:01:36.102UTC 생성됐다. 실제GPU가격0.59USD/h,
 운영상한0.65USD/h·총2.30USD·회수예약1USD·compute cutoff06:01:36.102UTC/
 job deadline05:59:36.102UTC를 고정했다. 재시도 시 최초 과금과 한도를 재설정하지 않는다.
- Pod가 MMDL_test의2b754a849e0abbb004ce984c40d49350918a2b83을 직접 clean clone했다.
 직접확인 A40/49140MiB·driver580.178.04·Python3.12.3·cgroup RAM49,999,998,976bytes,
 120GB/workspace mount다. 별도 Network Volume0이며 개인키·로컬 모델/venv·코드 overlay를
 업로드하지 않았다. GitHub clone 때120초 제한 agent forwarding만 사용했다.
- 명시목록385payload/2,064,900,296bytes·transport387개를 rsync로 전송하고 Pod에서
 모든SHA/크기/regular+nlink1/비symlink/파일목록/동일GitSHA·clean을 검증했다.
 closure SHA89c711a1…; verifier receipt는 runtime/remote-private-transfer-verification-v1.json이다.
 benchmark 제외 검사용 원장/이미지는 inference forbidden roots에 넣으며 학습에 쓰지 않는다.
- 현재 setup watchdog205/pip240가 실제 실행 중이며 설치 데이터를 계속 쓰는 것을 확인했다.
 764 CPU tests/doctor/CUDA BF16/모델13파일 검증을 포함한 정상 setup을 생략하지 않는다.
 13:14KST까지 모델 준비/실제 RL/체크포인트 reload/150평가는 아직 미실행이다.
 다음 계정조회는04:20:49UTC까지다. setup 최대1시간과 전체 금융·시간 watchdog은 유지한다.
- 실제 native RL/reload를 우선하고 남은 시간/비용으로 paired150 가능 여부를 판단한다.
 같은8K 조건의 F0/FT가 필요하며 과거32K558점과 비교해 개선을 주장하지 않는다.
 교수님MMDL/V8는 불변이다. 소유Pod의 결과 회수·검증 뒤 정확한Pod만 삭제한다.

## 2026-10-17 12:59 KST — full scan·직접 검토·제한 데이터 preflight 완료

- 실제100개 native Parquet/16,740행 인덱스 및 OCR/SSCD/SigLIP/calibration을 사용한
 12단계 전체 검사 COMPLETE/0blocker. 원본 receipt SHA
 `afcc05794bd718e4eb96411758c415ea61668e35349566ee9e14f10afd5a78ad`,
 1,966,628,861byte evidence SHA
 `ba8beec20fcf5b8ee1a5dd28edec0b86e42b6a0b0510a7f2129fda7541daa4ce`.
- 경고1953행/고유1303개 직접 AI 검토 결과 DISTINCT1953/duplicate0/unresolved0.
 기존1858 경고는 동일 입력/경고/증거 byte 결합 후 재사용하고 새OCR95건을 직접 대조했다.
 질문/이미지 exact, SSCD/SigLIP 경고는0이며 이를 미실행과 혼동하지 않는다.
 타일 보정 FPR90.625% 등 검출기의 한계를 보존하며 human_verified=false다.
- 원본 build와 모든 경고를 보존하고76개 입력이 동일한 별도 build-reviewed-v1의 metadata를
 명시적으로 재결합했다. 실제 approval worker106.933688초/VALIDATED_FOR_MEASUREMENT/
 0blocker로16train이 적격이고24heldout은 제외됐다. 2개 단일/복수이미지 probe도 같은
 적격 actor에서 생성했다. 최초 로컬 ActorItem 출력 직렬화 실패와 재검증을 구분한다.
 D1–D8/preflight/probe/summary는 비공개 candidates-v1/smoke-data-reviewed-v1에 있다.
 이는 <=2update program/AI 검토 smoke 적격성이며 일반 campaign·Phase A·GPU 성공이 아니다.
- test-only 게시 SHA2b754a849e0abbb004ce984c40d49350918a2b83의 worker module 경로를
 사용한 전체764 CPU tests318.928초 PASS(skipped1, Windows mount 전용 검사).
 시험 당시 Linux clone HEAD는f774e93이고 test파일 byte가 새 게시본과 같았다.
 임시 복사의 실행권한을644로 맞춘 후 최신 게시 tree와 전체 tracked diff0을 확인했지만,
 로컬 시험 파일 변경이 남아 있어 fast-forward는 Git이 거부했다. 이를 clean 새SHA 실행으로
 허위 표시하지 않는다. RunPod는 새SHA clean clone으로 모든 검사를 다시 수행한다.
 test harness 추가 mypy31진단은 미해결이며 production 범위 검사 PASS와 구분한다.
- 새 Pod0/cloud지출0. SECURE A40 최신 조회 가격0.59USD/h이며 실제 대여 전 가격/범위와
 USD2.30/회수예약1USD를 다시 결합한다. 실제4B RL/reload/MMMU150은 아직 미실행이다.
 교수님MMDL clean HEAD197838c11557911a14a46d87b7f52fda0cac5b3c·V8은 불변이다.

## 2026-10-17 12:25 KST — 검증된 중복 심사·ID 제외 코드 MMDL_test 게시

- 게시 SHA: `f774e9397c118bb9511b8bf9253b2da3ac365041`, `feat/workspace-repro`.
  원격/로컬 HEAD·clean과 명시28개 상위 파일 byte/mode 일치를 확인했다. 기존 Git author의
  계정 identity를 해당 commit 명령에만 적용했으며 전역 Git 설정은 바꾸지 않았다.
- 실제 원문 직접 AI 검토: 초기1858경고/고유1208개 모두 DISTINCT, 중복·미해결0.
  이미지218개·텍스트 참조11개를 확인한 범위다. 전수 데이터 무오염 인증이나 사람 검토가 아니다.
  검증 파일 `candidates-v1/overlap_review_v2/main_verification_v1.json`
  SHA `54f2c42a926ff954bf8290ff5ff50cb9c5025e0d0bb5a0a8cfed90eeb370a00b`.
- 원래 전체 index/input 및 모든 경고를 보존한다. 새 정밀 검사에서 추가된 후보는 별도 직접
  검토가 필요하며, 학습 전 complete scan/D6/엄격한 재결합을 통과해야 한다. blacklist는
  학습 ID/family를 제외하며 향후 MMMU 평가150의 분모를 바꾸는 기능이 아니다.
- 실제 SigLIP tokenizer에서 구두점 전용11필드가 content token0인 경계를 수정하고 native
  64길이 EOS/padding 결과와 일치함을 확인했다. 새 source578c3fda…에 연결한 OCR/SSCD/
  SigLIP 실행은 진행 중이며 과거 캐시를 새 producer 결과로 소급 재표기하지 않는다.
- 연결58 CPU 검사·게시본42 집중 검사·Ruff/범위 mypy·독립 delta 검토 PASS.
  이는 작은 CPU/fixture 증거이며 실제4B CUDA optimizer/reload나 성능 개선 증거가 아니다.
- 새 RunPod 조회0, 이번 cloud 지출0. 교수님 MMDL clean HEAD
  `197838c11557911a14a46d87b7f52fda0cac5b3c`와 V8 558/900은 불변이다.
  다음은 추가 후보 직접 검토→최종 입력/증거 preflight→USD2.30 내 실제 제한 GPU RL이다.

## 2026-10-17 12:06 KST — 사용자 목표 재개·직접 대조 및 blacklist 연결 중

사용자가 active 목표에 유사 경고의 실제 문제 대조, 동일 문제 blacklist/ID skip 후 진행을
명시했다. 아래 중단 정책을 대체한다. 원래 검출기/원장은 보존하고 별도 검토를 결합한다.
단순 경고를 동일 문제 확정으로 사용하지 않는다. 원장 재확인상 텍스트 경고는35/40문항,
이미지 경고는40/40문항이며 각각11개 benchmark 레코드와149고유 이미지에 연결된다.
실제 텍스트 사례의 공유5–7gram은 A B C D E F G 노드 문자 나열이었다. 전체 후보의 실제
직접 대조와 이후 전체 OCR/SSCD/SigLIP 원장 연결은 진행 중이다. 기존40개/60PNG 입력은 불변이다.

새 `adjudication.py`는 원본 전체 검사 완료, 후보별 SHA 및 근거 파일·AI 검토를 검증하고
DISTINCT만 해제한다. 원본 경고 목록은 보존하며 DUPLICATE/UNRESOLVED와 같은 family는
차단한다. exact raw/pixel 이미지와 source/split 위반은 해제할 수 없다. 제한 smoke의
preflight가 blacklist ID를 반환 actors에서 제외하며 전부 제외됐으면 실행을 거부한다.
ordinary training의 기존 승인 조건은 유지한다. 실제 GPU 성공이나150문항 완료는 아직 아니다.

현재 동결 엔진의 전체 OCR/임베딩과 실제 보정 자료를 로컬에서 마무리하고 있다.
이는 benchmark 제외 검사용이며 상위 모델 풀이·SFT 학습 자료를 생성하는 작업이 아니다.
새 cloud Pod/GitHub 게시 전이며 MMDL/V8는 불변이다. 모든 공개/실행 상태는 실제 완료 증거로 갱신한다.

## 2026-10-17 11:53 KST — 사용자 조건에 따라 중복 후보 검출 후 RL 진행 중단

사용자는 검출기를 유지하고 중복 후보가 나오면 중단하도록 명시했다. 후보를 통과시키거나
임계값을 완화하지 않는다. 제한 AI/program 검토 smoke와 A40 대체 승인도 이 중단 조건을
우회하는 권한이 아니다. 실제 pretrained4B RL·fresh GPU reload·MMMU150은 미실행이다.

- 원본100 Parquet/16,740행/이미지19,041회·고유12,168장의 전체 native 인덱스를 실제
  재구축했다. `exclusion-v2/index.json` SHA256은
  `82b1084d24bfd9bde5c287a15d5b5c2a1b493776597b6dd25cfb0b4e8526ed09`다.
  원문·이미지는 비공개이며 학습 데이터가 아니다.
- 고정40문항/60PNG·4split을 기존 검출기로 대조했다. 텍스트 n-gram 후보1,165건과
  perceptual 후보693건, 총1,858개 후보 기록이40문항 전체에 걸쳐 발생했다. 동일 문항의
  여러 후보가 포함되므로1,858문항이라는 뜻이 아니다. 질문 exact와 raw/pixel exact
  이미지 후보는0건이다. 유사 후보는 실제 복제 확정이나 거짓 양성 확정이 아니다.
- 실제 원장은 `candidates-v1/stop_on_candidates_native100_v1.json`, SHA256
  `d20b837316d714889a3b7b4bcc5f1d6c97f6e7a00441d1ca2758d916021f08f6`이다.
  상태는 `BLOCKED_NOT_CHECKED`, quarantine40/allowlist0이다. 텍스트/exact 이미지
  대조 범위는 완료됐으나 전체 OCR/SSCD/SigLIP/보정과 최종 source 권리 원장은 미충족이다.
  이를 전체 비중복 검사 통과로 표시하지 않는다. 후보 차단만으로 사용자 중단 조건이 충족됐다.
- 로컬 OCR은 소유 process group908654에 SIGTERM을 보내 종료했고 닫힌 캐시는 보존했다.
  SSCD/SigLIP 캐시도 부분 결과로만 보존한다. 보정 material/실제 보정 실행은 시작하지 않았고
  검출 임계·후보 해제 로직은 변경하지 않았다. native index 및 마지막 읽기 전용 대조도 종료됐다.
-40개 재생성·독립 solver 일치,600개 정답 변화 개입 쌍과1,300 PNG 검증은 별도 CPU
  증거다. `data_evidence_components_v2/components.json` SHA256
  `4274200a3f0474e4669f3225dd3621525ea43567d55eda2c332f312fd8e0d443`이며
  AI 검토는 `human_verified=false`다. D6 미충족이므로 학습 승인으로 승격하지 않는다.
- 구현에는 제한 smoke 분리, 실제 native index/엔진 캐시 생산, source/snapshot 변조 거부,
  정확한 inventory JSON 게시 허용 경로와 회귀를 추가했다. 이는 로컬 미게시 변경이다.
  승인 연결25검사와 엔진/보정/원장/게시23검사, Ruff·bounded mypy가 통과했다.
  최초 추가 index 테스트 명령의 잘못된 module 이름은 import 오류였으며 코드 결함 통과로
  간주하지 않는다. 올바른 실제 파일명으로 index/실측 component12검사9.949초 PASS를 확인했다.

증거 루트는 `/home/jang2/mmdl-artifacts/notebook-rl-smoke-20261017/`다. 이번 새 Pod 생성·
유료 실행·GitHub push는 없으며 최종 조회 Pod0이다. MMDL_test HEAD
`89271c277bbe45177e62396377a0084b1dde49b5`와 교수님 MMDL HEAD
`197838c11557911a14a46d87b7f52fda0cac5b3c`는 clean/불변이다. 기존 V8 558/900도 불변이다.
중단 결과를 학습 파이프라인의 실제 GPU 성공이나 성능 개선으로 보고하지 않는다.

## 2026-10-17 07:33 KST — 전체 GPU 검증 순서·저장 step1 캐시 대조 연결 게시

`gate_session` v2는 기존 G0–G13 측정 함수를 한 실행 순서로 연결한다. capture의
독립 F0/gradient 검사→engine 입력·token/logprob 검사→동일 shared update0→1→즉시
checkpoint 저장→동기화 후 검사→새 version1 batch/G12 update1→2 순서를 고정했다.
부모는 capture/replay/CPU export/default-cache/fresh-engine의 실제 5개 자식 프로세스를
순차 소유한다. 뒤 세 단계는 live step2가 아닌 저장 step1만 사용하며, 새 engine은 그
BF16 export에서 시작한다. BF16 변환에서 실제 Merger 변경이 사라지면 실패를 유지한다.
G12 실제 producer의 8개 policy/reference logits·logprobs 비교 결과를 기존 승인 계약의
필드에 연결했다. 성공한 scalar 검사도 승인서·학습 효과·Phase A 완료로 바꾸지 않는다.

- 작은 실제 Qwen/PEFT CPU export·checkpoint 재현·변조 거부와 mock 연결 회귀 등을 포함한
  루트 60검사/24.166초 PASS(게시 안전3 포함). 게시 checkout 연결 검사57개/25.569초 PASS.
  Ruff7파일·bounded mypy3파일 및 독립 Sol delta 검토 PASS다. 1-token probe를 허용하면
  GPU 적재 후 G9에서 실패하는 불일치를 CPU preflight의 2–32 token 제한으로 바로잡았다.
  첫 새 테스트의 예상 예외형 오류는 테스트를 실제 approval._Blocked 계약에 맞춰 수정했다.
  실패 조건이나 runtime 검증을 약화한 것이 아니다.
- 코드/시험/README 8파일만 MMDL_test feature branch에 정상 게시했다.
  commit `89271c277bbe45177e62396377a0084b1dde49b5`, 원격/로컬 HEAD·clean·상위8파일
  byte 일치 확인. 내부 보고서/입력/응답/가중치·무관한 dirty 변경은 게시하지 않았다.
  교수님 MMDL clean HEAD `197838c11557911a14a46d87b7f52fda0cac5b3c`·V8는 불변이다.

현재 구현과 CPU 시험은 완료된 이 연결 범위의 근거일 뿐, 실제 pretrained4B CUDA/RL,
모든 GPU gate의 실측 통과, approved pool/Phase A 또는 MMMU150 완료가 아니다.
사람의 D3 렌더·D7 누설 검토, full exclusion/OCR/embedding/calibration 및 실제 두 이미지/T3
조건은 아직 미충족이다. AI/program 검토를 사람 인증으로 쓰거나 실제 입력을 꾸미지 않는다.

22:30UTC경 live 조회: Pod0, COMMUNITY/count1/CUDA>=12.8의 RTX3090 재고 LOW로 변경됐다.
이는 할당 성공이 아니다. 정확한 A50 항목은 없고 A40는 이 COMMUNITY 조회에서 NONE이다.
지정 모델 safetensors8,875,719,344bytes 기준 두 복사본+고정10GiB KV만26.532GiB이므로
24GB3090은 현재 protocol의 사전 최소조건을 만족하지 않는다. G3 post의 별도 F0까지
세 복사본+KV는34.798GiB이며 optimizer/활성값 등은 추가된다. 48GB 실측 적합성도 미확인이다.
약2.30USD·회수예약1USD를 유지하며 A40 대체 및 사람 인증 없는 별도 제한 smoke에 대한
미확인 선택을 임의 승인하지 않는다. 새 유료 Pod/학습/API 호출은 시작하지 않았다.

## 2026-10-17 07:17 KST — 제한 RL 업데이트·새 프로세스 재검증 연결 게시

`fine_tuning/notebook_rl/gate_session.py`는 CPU 부모가 실제 두 자식 프로세스를 순서대로
소유하여 G4/G11/G13의 동일 update0→1과 G12 capture/replay를 연결한다. 첫 업데이트를
즉시 별도 checkpoint에 보존하고, 동기화된 version1에서 새 원본 이미지로 다음 batch를
수집한다. 다음 batch/RNG의 step1을 저장한 뒤 capture가 update1→2를 수행하며, 별도
fresh child가 같은 step1과 입력을 복원해 update1→2를 재검증한다. live step2를 step1로
내보내지 않는다. G13 관측 범위는 shared0→1이며 G12까지 관측했다고 하지 않는다.

- 데이터의 실제 ActorItem 형태와 canonical hash, 단일/두 이미지 probe의 eligible actor
  동일성을 검사한다. replay에는 모든 split의 private gold/lineage 접근 차단을 적용한다.
  frozen-weight 권한을 RL 권한으로 사용하지 않고 별도 bounded-update purpose를 요구한다.
  기존 D1–D8, 사람 검토, 오염 검사, exact checkout/lock/watchdog/메모리 조건은 완화하지 않았다.
- 실제 작은 CPU Qwen의 RL update와 owned subprocess timeout/cleanup을 포함한 연결 회귀
  56개/222.852초 PASS. 게시 checkout의 session/checkpoint/inputs 32개/20.802초 PASS,
  게시 안전 검사3개·Ruff3파일·bounded mypy2파일 PASS. 독립 Sol의 두 지적(실제 actor 객체,
  replay의 전체 split 차단)을 수정하고 첫 checkpoint 보존의 추가 delta 검토도 통과했다.
  합성/mock 및 작은 CPU 모델의 결과를 실제 pretrained 4B CUDA 검증으로 표시하지 않는다.
- 코드/시험/README 4파일만 MMDL_test feature branch에 정상 게시했다.
  commit `99007d7c867d83244dc1fb5c3c18bd0b7d8bb1f4`; 원격 SHA/로컬 HEAD 일치,
  clean 상태 및 상위 4파일 byte 일치를 확인했다.
  내부 보고서, private 입력/응답/가중치 및 상위의 무관한 변경은 게시하지 않았다.

이 명령은 G4/G11/G12/G13용 제한 측정 실행기이며 전체 G0–G13, Phase A, 실제 4B RL,
학습 효과 또는 MMMU150 완료가 아니다. checkpoint cursor의 NOT_TRAINING_RESUME는 범위
표시이며 generic checkpoint loader의 접근금지 장치라고 주장하지 않는다. 남은 구현은
나머지 GPU gate와 G6 step1 export/default/fresh 비교의 연결이다. 실제 approved pool과
2-image/T3 coverage·사람 검토·전체 exclusion 조건도 아직 충족하지 않았다.

22:12UTC RunPod 조회에서 Pod0, COMMUNITY/count1/CUDA>=12.8의 RTX3090 재고 NONE,
정확한 A50 항목 부재를 확인했다. 새 유료 자원은 생성하지 않았다. 약2.30USD 예산과
회수 예약1USD를 유지하며 A40 대체/AI·program-only 제한 smoke의 미확인 선택을 임의 승인하지 않는다.
교수님 MMDL clean HEAD `197838c11557911a14a46d87b7f52fda0cac5b3c` 및 V8 결과는 불변이다.

## 2026-10-17 06:53 KST — 오답노트 RL G12 입력 전달 연결·MMDL_test 게시

활성 목표는 내부 v2 오답노트 RL 구현·실제 제한 예산 RunPod 검증이다. 별도 연구 보고서
`claudedocs/research_verified_self_correction_rl_20261016.md`의 독립 episode/G4 설계와
v2의 T1:8/T2:4 조건을 섞지 않는다. 이번에는 실제 GPU 학습을 완료하지 않았다.

- `fine_tuning/notebook_rl/gate_inputs.py`가 다음 policy-version의 실제 트리/보상/토큰,
  원본 이미지 byte SHA와 순서·decoded pixel, probe seed/cap 및 cursor를 비공개 write-once
  packet으로 연결한다. message probe는 최초 전처리의 literal token으로 보존하며 원요청은 불변이다.
  capture/replay wrapper가 완료 capture hash에 packet을 연결하고, RNG 복원 전에 이미지를 재구성한다.
  replay InputPreparation.policy_version은 저장 step과 일치시킨다. CPU 시험용 packet은 승인서가 아니다.
- 실제 작은 Qwen CPU 모델의 두 독립 프로세스에서 packet을 재사용해 정책/reference 전체 completion
  logits, 다음 RL 손실, 업데이트 후 모든 parameter/optimizer 상태와 cursor 일치를 확인했다.
  큰 모델의 자기수정 학습이나 의미 성능 개선을 검증한 것은 아니다.
- 연결 회귀 41개/176.965초 PASS. 최종 추가 wrapper/예약 context 검사를 포함한 새 입력 검사8개 PASS,
  최종 MMDL_test checkout의 입력+G12 검사18개/18.306초 PASS, 게시 검사3개·Ruff4·mypy2 PASS.
  독립 Sol이 예약 context, 새 source 의존성, wrapper 양성시험 누락을 지적했고 모두 수정·delta 재검토했다.
  최초 새 fixture의 usable group 단일 샘플 오류는 fixture를 2샘플로 바로잡았으며 runtime guard를 완화하지 않았다.
- 위 코드/시험/README 5파일만 MMDL_test `feat/workspace-repro`에 정상 게시했다.
  commit `6395278ef07643ab209626ae75d3cbfe0cb8fa0b`; 원격 SHA/로컬 HEAD 일치·clean 및
  상위 5파일 byte 일치 확인. 원본 내부 보고서/전체 dirty worktree를 일괄 게시하지 않았다.
  교수님 MMDL은 clean/HEAD `197838c11557911a14a46d87b7f52fda0cac5b3c` 불변이다.

남은 실행 연결은 model-free parent가 실제 direct Popen들을 소유하고 G0–G13을 순서대로 묶는 것,
G11의 shared update0→1과 별도 version1 batch/G12 update1→2를 분리하는 것, step1 export로 G6
default/fresh 캐시 비교를 닫는 것이다. 현재 G11은 2이미지·실제 T3를 요구하지만 pilot은 모두1이미지다.
이 coverage를 꾸미거나 정답을 강제로 재시도시키지 않는다. G3 POST_SYNC는 정책/새 F0/엔진 세 모델이
동시에 필요하므로 기존 2-copy+10GiB KV 사전 floor를 실제 peak 적합성으로 보고하지 않는다.
전체 GPU gate/완전 Phase A/실제 approved pool/4B RL/새 MMMU150은 미완료다.

이번 RunPod 실제 목록은 Pod0이다. COMMUNITY/count1/CUDA>=12.8 catalog의 RTX3090/A40는 NONE이며
정확한 A50 entry가 없다(A5000와 혼동하지 않는다). 유료 자원을 생성하지 않았다.
기존 약2.30USD 범위를 유지하며 A40 대체와 사람 검토 없는 제한 AI/program-only smoke의 명시적 선택은
아직 확인되지 않았다. D3/D7 사람 검토·전체 benchmark exclusion/OCR/embedding 조건을 소급 완화하지 않는다.

## 2026-10-12 04:25 KST — 고정 진단·native A/B 실험 종료, 전체 회수·과금 자원 삭제

아래 pending 스냅샷보다 우선한다. c551 진단336와5093 A/B 학습·외부396을 실제 완료했다.
새 setup223+24 CPU tests/lock/pip check/실제CUDA·BF16/공식13파일 hash를 통과했다.
모든 stage/outer는 exit0이고 원문732개·학습 입력·가중치·로그를 로컬 검증했다.
교수님MMDL/V8 558/900은 변경하지 않았다. 새900 평가나 전체 모델 개선의 완료가 아니다.

### 1. 입력 정보 회복 진단 — 새8K, 선정48문항

| 조건 | 제공 정보 | 정답 / 48 |
|---|---|---|
| I | 이미지 | 22 |
| T | 시각 전사 | 24 |
| IT | 이미지 + 같은 전사 | 26 |
| IR | 이미지 + 일반 규칙 | 20 |
| TR | 전사 + 같은 규칙 | 22 |

I→T 회복7/악화5, I→IT 회복8/악화4다. 동일12 I/T 정답은
F0 4/3, FL5/5, FM3/4, FV2/4, FML5/3이다. 한 seed·작은 선정 표본이며
가족 단위 신뢰구간에0이 포함된다. 내부 모듈 책임이나 MMMU900 정확도로 일반화하지 않는다.
전사48개는 메타데이터상 COMPLETE/VISUALLY_INSPECTED, human_verified=false다.
IR/TR 규칙 감사 PARTIAL33/UNRELATED15·충분성 인증0은 전사 분류가 아니다.
따라서 TR 실패로 규칙 적용 능력 부족을 확정하지 않는다. 선언된 전사 완전성도 사람 인증은 아니다.

### 2. teacher-free native 감독 비교 — freshF0, 고정FM

A는 최종답, B는 원본 표의 순서 있는 facts+같은 답을 감독했다. native1600과
동일TRACE360/NVW40을 합쳐 각각2000행·125update/학습률·순서·seed를 맞췄다.
감독token22061/127665·token-normalized retention loss의 상대 가중치는 같지 않다.
MMMU 파생 학습/상위모델 생성 풀이를 쓰지 않았다. 각 학습/fresh reload/export 완료 및
24개 FP32 Merger tensor/109105152 parameter의 key/shape/finite/hash 검증도 PASS다.

외부132(QA48/facts24/retention60)를 F0/A/B 각각 같은8K/vLLM0.11.0/동시4·입력·seed로 평가했다.

| 고정 채점 지표 | F0 | A: 답 | B: 표facts + 답 |
|---|---|---|---|
| QA 정답 / 48 | 35 | 30 | 28 |
| 표전체 exact / 24 | 19 | 11 | 17 |
| 표 행 exact / 24 | 19 | 16 | 23 |
| 표 셀 일치 / 310 | 268 | 253 | 299 |
| retention QA / 12 | 4 | 4 | 5 |
| retention TRACE / 24 | 8 | 11 | 11 |
| retention structure / 12 | 0 | 0 | 0 |
| retention grounded / 12 | 3 | 2 | 0 |

세조건 모두 facts JSON/schema24 유효이므로 B의 행/셀 증가는 단순 JSON 유효성 증가가 아니다.
B는 행4회복/0악화·셀+31이지만 제목/헤더 손실로 표전체는 감소했다. QA는 A+7/−12,
B+7/−14이며 B에는 facts만 반복/출력한 length3과 자연 종료 답 미완성이 있다.
표24는 QA 일부와 동일한 표이므로 추가 독립 QA24로 세지 않으며310셀도310문항이 아니다.

정답을 배제한 AI 원문 감사 후 별도 host 비교 민감도는 F0/A/B39/41/30 of48이다.
표현 누락을 포함한 탐색 지표이며 고정35/30/28·사전 확대gate를 바꾸지 않는다.
B 감사39 CLEAR/1 AMBIGUOUS/8 NO_FINAL; 사람 의미 인증/새 제출 성적이 아니다.
관측은 “B의 행·셀 보존 개선과 최종답 악화가 함께 발생”이며 단일 모듈 실패의 증명은 아니다.

### 3. 결정·회수·삭제·다음 단계

사전 QA 회복·표전체/셀 개선·retention 유지의 gate는 A/B 모두false다. 50K/새 모듈 sweep/
최종8K900을 자동 시작하지 않는다. 다음 프로토콜은 답 형식/채점을 추론 전에 함께 동결하고,
facts-only와 QA-after-facts 과제를 분리하며 감독token/retention 가중치·독립 family/seed 반복을
통제해야 한다. 규칙/중간 상태 C는 검증된 native/program 감독을 확보한 후의 별도 조건이다.
현재는 다음 설계 제안이지 실행/효과 검증이 아니다. 과거원문·gate를 소급 변경하지 않는다.

834 닫힌 payload·2103 private 입력/이미지·원문336+396·두 전체 학습 checkpoint·환경/로그를
결합 검증했고 독립delta PASS/blocker0다. final_local_closure SHA935ed744…와
tabmwp_final_delta_independent SHA0ba1a3c6…에 연결했다.19:24:54UTC 소유Pod z2sa5xt3b93er9
삭제204→19:25:01UTC get404/listpods0/networkvolumes0이다. 원격container30GB/workspace120GB는
복구 불가이나 로컬 결과는 보존했다. 과금 구간 약4시간38분·GPU 단가0.59USDh 추정2.73USD,
운영상한0.65USDh 기준3.01USD이며 API snapshot2.1842USD는 최종 청구서/잔액이 아니다.

결과/검증은 `analysis_exports/input_recovery_20261011/`의
`full336_scores_v1/`, `F0_A_B_external396_scores_v1/`,
`F0_A_B_QA144_semantic_audit_sensitivity_v1.json`, `external396_fact_field_controls_v1.json`,
`final_scientific_decision_v3.json`, `final_local_closure_v1.json`,
`owned_pod_cleanup_final_v1.json`(86f0f5a9…), `final_cost_report_v1.json`(b23bb7f6…)이다.
해석 v1/v2의 잘못된 규칙/전사 축약은 보존하고 v3가 대체한다. frozen 점수·원문·입력은 불변이다.
가중치는 `/home/jang2/mmdl-artifacts/recovered/tabmwp-A-2000-20261012-z2s/`와
`/home/jang2/mmdl-artifacts/recovered/tabmwp-B-2000-20261012-z2s/`에 보관했다.
원격실행 SHA5093dc86536c68cc352cb34a0f2dd53db99c6519는 문서 게시와 별개로 불변이다.
MMDL_test에는 코드와 종료 기록만 게시하며 비공개 원문/이미지/전사/가중치를 올리지 않는다.
교수님 MMDL clean HEAD197838c11557911a14a46d87b7f52fda0cac5b3c는 그대로다.

## 2026-10-12 03:18 KST — 새 외부 F0 132 완료·A 실제 학습 진입

최종96/10단계/full336 독립검토까지 PASS다. 새5093 정상 setup은223 학습+24 진단 CPU
tests·exact lock/pip check·실제CUDA/BF16·지정13 모델 파일 검증으로 exit0 종료했다.
같은소유Pod/닫힌336/현재입력/새setup 결합 authority 후17:59:58UTC controller6275를
시작했으며 실행GitHub checkout5093은 이후 문서 commit과 별개로 고정한다.
외부F0 132개가 exit0 종료했고138 payload 전후SHA/요청/원문token decode·native8K/동시4/
입력ID·CPU-reference tensor 결합과 독립 CPU 재채점이 PASS다. QA35/48·표전체19/24·
셀268/310, retention QA4/12·TRACE8/24·structure0/12·grounded3/12이며 MMMU 점수가 아니다.
18:11:34UTC A가 freshF0/고정FM 학습에 진입했고18:16:33UTC432/2000행·27/125update를
실측했다. B/새396/학습개선/확장/최종900은 아직 미완료다. engine pixel·사람 의미 인증을
새로 했다고 주장하지 않는다. native 답/표 감독만 사용하며 teacher 풀이를 사용하지 않는다.
원금5.20USD·reserve1USD·상한0.65USDh/원과금시각 유지, GPU cutoff21:12:32UTC와
금융 deadline21:14:32.577UTC는 불변이다. 미확인 약속 충전은 예산에 더하지 않는다.
교수님MMDL·원래V8/558점은 불변이고 두조건 QA/facts/retention gate 후에만 확대를 판단한다.
근거 `analysis_exports/input_recovery_20261011/`의 `checkpoint96_independent_v1.json`,
`tabmwp_setup_closed_v1/`, `tabmwp_ab_launch_v1.json`, `F0_external132_closed_v1/`,
`F0_external132_independent_v1.json`(60f03f32…), `F0_external132_scores_v1/`,
`tabmwp_A_training_actual_start_v1.json`이다. 이전 pending receipt는 당시 상태로 보존한다.

## 2026-10-12 02:51 KST — 고정8K 진단336 완료·새 setup 진행

소유A40/c551의10단계가 전부 exit0으로 닫혔다. 336 원응답과 실행기록378 payload 파일을
로컬 회수·SHA/요청/record/V8 CPU채점·동일12 전이 산술까지 검증했다. 전체 점수 원장은
`analysis_exports/input_recovery_20261011/full336_scores_v1/`다. F0 48개 I22/T24/IT26/
IR20/TR22이며 동일12 I/T는 F0 4/3, FL5/5, FM3/4, FV2/4, FML5/3이다. 작은 균형 표본/
한 seed/부분 AI전사·규칙이며900 성적이나 내부 모듈 책임을 확정하지 않는다.
최종96 독립검토는 진행 중이다. CPU-reference와 실제 engine pixel tensor, BF16 export와
과거FP32 master 추론을 구분한다. read-only SSH 지연 한 연결은 안전하게 종료·재접속하고
회수했으며 GPU/원격 결과는 변경하지 않았다.

기존 controller 및 GPU child 종료 확인 후17:44:31UTC clean5093의 정상 setup PID5595를
시작했다. lock/223+24 CPU검사/doctor/공식 pinned model 취득을 거치며 아직 새 학습은 없다.
teacher-free native1600+retention400 A/B와 외부132×3은 CPU 승인/독립검토 완료 상태다.
답만 vs 표facts+답을 freshF0·고정FM·125update로 비교하되 target token과 retention 상대
gradient 가중치의 차이를 한계로 기록한다. QA회복·실제facts exact/cell recall·retention 유지
조건을 통과하기 전 확장/최종900을 자동 실행하지 않는다. 다음 authority는 최종336/새
model-ready/동일input을 확인한 뒤 발급한다. 아직 GPU 학습/396 외부평가 완료가 아니다.

원래21:12:32UTC cutoff와 최초5.20USD를 유지한다.17:48UTC 계정 RUNNING/$0.59h,
새 CPU 검사 중 GPU0%/0MiB를 확인했고 다음 계정 조회는17:58:43UTC다. 교수님MMDL과
원래V8/558점은 불변이다. 근거 `full336_main_verification_v1.json` SHA81490d27…,
`diagnosis_controller_closed_v1/local_receipt.json` SHAec992c6f… 및
`tabmwp_setup_launch_v1.json` SHA67e74abb…를 내부 증거 루트에 보존한다.

## 2026-10-12 01:23 KST — CPU 자료 승인·게시·회수 준비

원래 c551/소유A40 진단은16:20UTC F0 full110/240·GPU91%로 계속된다. 아직336 전체
완료가 아니며 새 모델 학습/전체900 점수는 없다. 새5093dc86536c68cc352cb34a0f2dd53db99c6519는
MMDL_test feature branch·원격HEAD/clean·별도Pod GitHub clone을 확인했다. 교수님MMDL
HEAD197838c11557911a14a46d87b7f52fda0cac5b3c와 V8/558점은 불변이다.
223 CPU tests532.436초·27 IR/게시검사81.207초·Ruff/mypy5/Bash PASS다.

TabMWP filtered trial35e547a564…의 실제A/B2000×2 processor/mask/EOS·pixel parity·
전체 제외 검사에 quarantine0이고 CPU DATA_APPROVED다. 감독 token A22061/B127665로
일치하지 않으며 행/seed/순서/125update만 맞춘다. source-trust만 승인했고 사람 인증은 아니다.
외부132(48QA/24facts/60retention) cohort58d5cc9460…와 별도 로컬gold를 동결했다.
Pod에는 gold 없이2103파일105685580bytes만 보내고 SHA를 전부 대조했다.

근거 `analysis_exports/input_recovery_20261011/{tabmwp_cpu_preparation_main_v1,`
`tabmwp_code_publication_v1,tabmwp_private_transfer_v1,monitor_20261011T1620Z}.json`.
독립 입력 검토는 진행 중이다. 새 환경 setup/학습 authority는 실제336 종료 후에만
실행한다. 기존 checkout/venv hot patch와 자동50K/900은 금지한다. 원래 GPU cutoff
21:12:32UTC는 실제 추가 잔액이 확인되기 전까지 유지하며 다음 계정 점검은16:30UTC다.

## 2026-10-12 01:00 KST — smoke60 로컬 검증·full 계속

01:03KST 추가: filtered native1600/200dev/200sealed·2000PNG 정상 취득 및 새1600
후보0 scan을 완료했다. 과거3후보 family와 옛 입력은 불변으로 보존하고 표본 선택은
모델 결과를 보지 않은 동일 seed 정렬을 사용했다. source/image 준비 hash1bedde7032…,
scan SHA4e2adfd591d06521255d09086f00bc09b9f33b467c002d827dd8e0a4f8075f85,
증거 `tabmwp_filtered_preparation_v2/`와 `tabmwp_native1600_candidate_scan_v2.json`.
58 연결 검사·14 source/image 검사/Ruff/mypy5 PASS. 현재 normal2000 mixture 준비를
실행 중이며 이후 실제 processor4000·CPU승인·외부 cohort를 결합한다. 학습은 아직 없다.

F0 smoke60은15:54:47UTC exit0·1761.693초로 완료했고 동일 출력의 full240으로 이어졌다.
닫힌60응답/manifest/backend를 `analysis_exports/input_recovery_20261011/`
`F0_smoke60_closed_v1/`에 회수해 원격/로컬 SHA·요청·record hash를 검증했다.
receipt SHA0ad69d98d83e6df8edf0a09a72e7c775af6654bb71bd3ef57f511150d638112d.
`F0_smoke60_scores_v1/`는 전체336 INCOMPLETE이며 완료된12×5 조건만 관측했다.
원문/seed/출력8K/동시4는 불변이며 새MMMU900/학습개선/Pod삭제 완료가 아니다.

TabMWP 후보3개 전체 table-family 사전제외와 immutable scan 결합을 구현했다.
같은 quota1600/seed·기존 dev/sealed·옛 준비본을 보존한다. 연결58 CPU검사67.422초/
Ruff PASS, helper31검사11.402초/bounded mypy4 PASS. 새 자료 준비와 실제 processor4000/
전체 제외·2000×2승인은 진행 전이며 새 코드/교수님MMDL/실행 c551을 혼동하지 않는다.
추가 승인을 반복 요청하지 않지만 확인된5.20USD 기준 원래 금융기한을 유지한다.

## 2026-10-12 00:50 KST — 실제 GPU 진단·사전 학습자료 검사

15:45:41UTC 소유 A40/c551의 F0 smoke45/60 및600초 HEALTH_POLL·GPU94%를 확인했다.
중간 닫힌48응답/manifest/backend를 `analysis_exports/input_recovery_20261011/`
`F0_prefix_20261011T1548Z/`에 회수하고 원격 SHA를 전부 대조했다. 별도
`F0_prefix_scores_20261011T1548Z/`는 INCOMPLETE·48실호출·조건 정확도null이며
기존900/62% 또는 full336 성적으로 사용하지 않는다. 원래 financial/GPU cutoff는 불변이다.

TabMWP order-bound v2의 actual2000행·동일 native1600/retention400과 실제 trainer 순서
SHA를 검증했다. 기존3회귀는 controller 합성 검사이며 순서 전용5검사와 구분한다.
현재 신규28 CPU검사9.508초·Ruff·mypy3 PASS이고 실제 새학습 인증은 아니다.
native1600 전체 fingerprint scan에서 유사 이미지 후보3개를 확인했다(exact/neartext와
exactimage0). 실제 benchmark PNG 감사는 접근 가능한32/100파일·2630/16740행만
새로 읽었으며 68개 미접근을 명시했다. 어떤 후보도 승인예외로 통과시키지 않는다.

학습 전 보수적 source revision으로 후보3개(14424/1148/29175)의 table family를 제외하고
원래 seed 정렬로 native1600을 다시 동결하도록 구현한다. 과거준비/scan/AI권리review는
불변으로 남기고 새상태를 별도로 기록한다. 진단표본/이미지/규칙/V8/교수님MMDL은
변경하지 않는다. 아직 실제 processor4000/정상2000×2승인/새 GitHubcommit/새학습/
132×3 외부응답은 미완료다. 상위모델풀이·감독은 사용하지 않는다.

## 2026-10-12 00:30 KST — 설치 완료·첫 GPU 진단 응답 확인

exact c5513ae 소유 A40의 setup1944.236초/exit0·192+24 CPU검사·실제CUDA/BF16·
모델13파일 hash를 확인했다. setup4파일과 로그를 로컬 회수했고
`analysis_exports/input_recovery_20261011/setup_closed_v1/local_setup_verification.json`에서
원래 manifest·실제 검사/exit0·파일SHA를 검증했다. doctor의 host RAM/공유df는
Pod cgroup 약50GB/120GB 작업볼륨 quota와 구분한다.

controller1178/child1182는15:25:26UTC F0 smoke60을 시작했다. 15:30UTC 첫2응답
stop과 실제 A40 GPU95%/40269MiB를 관측했다. 동시4·BF16·8K 및원래 cutoff를 유지한다.
원문 응답은 `/workspace/ir-runtime/runs/input_recovery/F0/`에 저장되고 stage로그는
`/workspace/ir-control/diagnosis/F0_smoke.log`다. smoke완료/전체336/새score/최종회수/
Pod삭제는 아직 아니다. 처음12문항의60응답은 이후48문항 조건에 재사용한다.

다음 A/B 비교는 기존 train/audit/export/controller/evaluator를 재사용해 준비한다.
native1600+동일retention400, freshF0/FM/seed3407/1epoch/125update다. QA48+표구조24+
기존retention60의 고정외부132개/모델을 사용하며 primary retention은 QA12/TRACE24를
동일가중해 계산한다. 두조건은 감독토큰량이 다르므로 행/updates 일치와 구분한다.
문항 순서 hash를 실제trainer와 일치시키는3회귀(2.124초)/Ruff를 통과했다.
이는 합성 controller검사이며 실제CPU 데이터승인/새학습/성능개선은 아직 아니다.
실행중 c551은 변경하지 않고 새 GitHub commit/cleanclone으로 다음단계를 진행한다.

## 2026-10-12 00:05 KST — 고정 GitHub clone·비공개 전송 검증, 환경 설치 중

코드 `c5513aef6c6994f15a37301c5b2a0f8e8bde74aa`는 MMDL_test feature branch에
정상 push됐고 로컬/원격 SHA·clean checkout·684파일 allowlist가 일치했다.
소유 A40 `z2sa5xt3b93er9`가 exact commit을 GitHub SSH로 직접 받았다.
HTTPS 인증 실패는 개인키 업로드 없이 120초 SSH agent forwarding으로 해결했다.
RunPod 볼륨의 chown 금지로 첫 rsync가23으로 끝났지만 owner/group 보존을 제거한
재전송 후 실제54입력파일과 네 checkpoint4/2/4/5파일의 전체 SHA256·목록 일치를
검증했다. gold.jsonl/코드 overlay는 보내지 않았다. 원본 checkpoint도 불변이다.

실측 A40 46068MiB·Python3.12.3·CPU RAM약50GB이며 설치PID265/299가 생존했다.
고정 lock 설치는 진행 중이고 GPU는 아직0MiB/0%다. 환경 준비와 실제 추론을
혼동하지 않는다. 설치 watchdog1시간 후192+24 CPU검사·CUDA/BF16·모델13파일
hash 검사를 통과해야336호출 controller를 시작한다. 동시4개·8K·원본 해상도 유지다.
실제 가격0.59USD/h·운영ceiling0.65/h·원래 잔액5.20/reserve1·최초과금14:46:51.039UTC,
금융종료21:14:32.577UTC/GPUcutoff21:12:32UTC는 변경하지 않았다.

독립 gold-blind 규칙 감사48개는 PARTIAL33/UNRELATED15, 충분한 커버리지 인증0이다.
따라서 IR/TR 전체 평균을 완전한 knowledge oracle로 해석하거나 TR 실패에서
규칙 적용 능력/모듈의 인과 책임을 확정하지 않는다. 전사도 AI 검토이지 사람 인증이 아니다.
I/T/IT 정보 회복과 규칙 힌트 민감도를 별도로 보고하며 frozen 입력은 소급 변경하지 않는다.

TabMWP 신규 변환기의7 tests/Ruff/mypy 및 독립 native2000쌍/manifest 재현은 PASS다.
외부 데이터 A/B1600train·200dev·200sealed는 teacher/solution을 쓰지 않으며 source table
family 기준813행을 제외했다. 실제 이미지 취득·학습 승인은 아직 완료되지 않았다.
새 학습은 현재 controller가 자동 시작하지 않는다. 교수님 MMDL과558/900은 불변이다.

추가15:14UTC 검증: PNG2000개/26,298,935bytes·error0의 취득과 독립2000개 hash/
decode/크기/pixel/역할 replay를 완료했다. Git tree와 공식 source14파일 결합도 PASS다.
취득본은 Linux private TabMWP root의 `selected_images_v1/`이며 receipt SHA
`be49f6891d8376e7d72832ae91a3ac147f08022757613123936caeca3c29f8bb`다.
신규11 CPU tests0.620초/Ruff/mypy2 PASS이며 실제 학습/정답 의미 인증은 아니다.
원래 취득/준비 false 승인 플래그는 소급 수정하지 않는다. 학습 후보1600native+400동일
retention과 외부 QA/표 구조 검증을 기존 train/audit/evaluator 재사용으로 준비한다.
RunPod는15:14UTC까지 설치PID299 생존·persistent volume I/O 대기와 실제 누적 write를
관측했으나 GPU 응답은 아직 없다. 원래 설치1시간/비용 마감과 실행 commit은 불변이다.

## 2026-10-11 — 새 입력 회복 실험 구현·CPU 검증 중

첨부 계획의48문항5조건 및 기존4개 checkpoint의12문항 I/T, 최대336호출을
`experiments/input_recovery/`에 구현했다. 원래 vLLM continuous scheduler와
checkpoint 복원·Controller·금융 watchdog·V8를 재사용한다. 신규8K protocol의
동시4개만 허용하며 기존 baseline 실행 조건/파서/558점은 변경하지 않았다.
정답표는 inference Pod에 보내지 않고 로컬 CPU 채점에서만 읽는다.

고정 입력 `analysis_exports/input_recovery_20261011/prepared_gold_bound_v2/`는
48개(A/B/C/D 각12, 과거정답6/오답6), checkpoint subset12 및 실제이미지51개를
검증했다. selection hash는
`562729cae39c13ff35a3fb08fef000cf56986b85bc5ee37bb78afcd20335550b`다.
첫 prepared bundle을 덮어쓰지 않았고 새 bundle의 selected gold까지 hash로 결합했다.
전사 AI검토/일반 규칙의 비완전성·G6 keep/loss의 제한된 커버리지를 공개 코드에도 명시한다.

선택/정답 변경, 모델 경로, 재개 원장 변조, 미완료/N/A 분모, checkpoint 전체복원,
동시 refill 및 deadline 검사를 독립 Sol이 검토했다. focused29 CPU tests PASS이며
전체 fine_tuning192개(378.371초), 신규실험24개(61.345초), 기존평가180개
(96.307초) PASS 및 Ruff/Bash/실험10파일 mypy PASS를 확인했다. 최초 타입 검사의
subset annotation/동적 PEFT merge 식을 보완했고 실제5개 복원 검사를 다시 통과했다.
MMDL_test 동기화·diff 검토 후 게시한다.
실제4B export·새GPU 추론·학습/새900은 아직 미실행이다. 합성 테스트를 실제 완료로
표시하지 않는다. 교수님 MMDL/원격은 불변이고 아직 새Pod를 생성하지 않았다.

RTX3090 품절/A40 48GB secure Low·GPU0.35USDh를 실시간 확인했다. 배포 직전 재조회한다.
잔액5.20USD·회수예약1USD와 저장비 포함 보수적 ceiling을 실제 과금 시작에 결합한다.
추가 충전 예정으로 한도를 자동 연장하지 않는다. 사용자 일괄 승인 범위 안에서
외부 원래 주석/프로그램 검증 기반 teacher-free A/B SFT를 이어가되 결과 gate를 유지한다.
MMDL_test exact commit 게시→Pod clean clone→설치/검사→60 smoke 포함336→회수/CPU채점,
이후 관측에 근거한 학습 비교와 외부 retention gate가 남아 있다.

## 2026-10-05 16:16 UTC — 최종 네 조건 분석·PPT 반영 및 보존 검증 완료

아래 15:05UTC의 PPT 미저장 변경 차단은 사용자 저장 후 해소됐다. 저장본 SHA08702d…를
먼저 대조하여 사용자38번 자막 위치 변경을 확인·보존한 뒤, 별도 작업본에서
19/27/31/33/34/41/42만 갱신했다. 최종 PPT는42장, SHA
`bedf8028ac43fc6604b43f95ec33942fd52a06bbdc0ad919d5a3cbd4e58ff008`이다.
원래1–18장 slide/notes/media78개 part, 나머지35장 slide/notes XML, 사용자38번 수정과
내장P1캡처10개 hash 보존을 검증했다. 수정7장 native1600×900 렌더·overflow0 및
주 에이전트/독립Sol 실제 시각 검토도 PASS다. 원본·사용자 저장본 백업과 실패 시도는
`analysis_exports/ppt_updates_20261005/`에 보존하며 과거 receipt를 덮어쓰지 않았다.
최종 증거는 `verification_closed10k.json`과 `final_closed10k/`의
`native_verification.json`, `visual_review.json`; 사용자 저장본 대조는
`user_saved_source_review.json`이다.

PPT에는 동일4K MMMU60의 F036/FL22/FM27/FV21/FML23, 외부 사전지표에 따른 FM 후보,
모든 조건의 실제rule-present0/2·전체structure exact0/12, 관측 한계·근거·HW·일정을
반영했다. 결과 분석 정본은 `claudedocs/g6_full10k_final_analysis_20261005.md`다.
결과 회수·CPU검증·소유Pod 삭제·이번 PPT 반영은 완료했지만 모델 개선 성공,
G7/독립 전문전이/50K/새8K·900 평가 완료가 아니다. Pod 삭제 증거는 기존
`analysis_exports/g6_runtime_20261005/pod_cleanup_receipt.json`에 그대로 있다.
이번에는 새GPU/Pod/유료API/외부 업로드/GitHub 게시를 시작하지 않았다.

## 2026-10-05 15:05 UTC — 네 full10K 회수·검증·Pod 삭제 완료, PPT 사용자 미저장 변경 보호

원래 네 조건 FL/FM/FV/FML은 각10000행/625update 및16개 train/reload/MMMU/external
단계 종료0,8개 paired input gate PASS로 완료했다. FML native SCP/528tensor CPU검증까지
완료한 뒤 전체controller/status/journal/source/outerlog/report와 기존 arm receipt를
결합한 최종 local gate를 통과했다. 증거:
`analysis_exports/g6_runtime_20261005/final_closed_controller/local_receipt.json`, SHA
`2ec35018f169710eb17ae7a4d093109d1a06ae31aa41cdc395698263b969d51c`.
FL의 과거 training receipt에는 pod_id 필드가 없어서 이미 byte 결합된 aggregate 및
세 remote receipt의 정확한 Pod/commit/run identity를 사용했다. 원본 receipt는 불변이다.
소유Pod22u0uqi171v0cn는14:57:07UTC 삭제204,14:57:20UTC 재조회404/Pod목록0/
NetworkVolume목록0으로 정리했다. 별도 NetworkVolume은 미생성이다. 필요한 결과와
네 전체checkpoint는 Linux recovered root에 보존했고 Windows에는 비가중치 원장/receipt를 둔다.
원격30GBcontainer/160GBPodvolume는 폐기돼 복구할 수 없지만 검증된 로컬 결과는 보존됐다.
정확한 삭제 증거는 `analysis_exports/g6_runtime_20261005/pod_cleanup_receipt.json`이다.

동일4K MMMU60 F036/FL22/FM27/FV21/FML23, 각gain/loss5/19·2/11·0/15·6/19다.
외부 개발 주지표 사전 규칙은 FM25/48>FML21/48>FL18/48=FV18/48(F015/48)이며
FL/FV 동률은 보조4/24>0/24로 구분한다. MMMU는 선택에 쓰지 않았고 유지위험은 별도보고한다.
전체structure exact0/12 및 실제rule-present0/2는 모든 조건에서 같아 전문규칙 학습 성공이 아니다.
최종 분석은 `claudedocs/g6_full10k_final_analysis_20261005.md`에 저장했고 독립 raw 감사와
26개 실재 근거를 연결했다. 비용은 실제13h01m59s×0.22/h≈2.87USD, 운영ceiling0.30/h≈3.91USD
추정이며 실제청구/현재잔액 조회가 아니다. 새 G7/50K/8K/full900/Pod/유료API/GitHub 게시는 없다.

PPT19/27/31/33/34/41/42의 최종 내용은 내부JSON에 준비했다. 독립 검토된 updater는
별도작업본 수정·native렌더·미변경ZIP part검사 후 atomic교체하고 사용자Saved/hash를 보호한다.
실제 실행은 COM의 Saved=0(42장, 파일SHA57ff8e…불변)을 확인해 첫guard에서 중단했다.
`final_closed10k` output도 만들지 않았으며 사용자의 열린 PPT/저장 파일을 수정하지 않았다.
원래1–18장/10개P1캡처/기존900 baseline은 불변이다. 최신 PPT 최종 반영·native렌더/독립시각
검증은 사용자 미저장 변경을 저장·대조한 뒤 남은 작업이며 완료라고 표시하지 않는다.

## 2026-10-05 14:24 UTC — 네 full10K 학습·평가 종료, 최종 회수 중

원래447c0fb controller는14:14:21.140979UTC에 FOUR_ARM_PILOT_EVALUATED로 종료했다.
네 조건 모두10,000개/625 update이며 FML MMMU60·external60 EXITED0과 두 input gate PASS를
원래 journal에서 확인했다. outer watchdog도 원래 금융기한 안에서 EXITED0이다.
14:23UTC API의 소유Pod22u0uqi171v0cn는 RUNNING/cost0.22USDh지만 GPU util/memory util0이다.
14:24UTC 실제 SSH의18개 알려진 PID 검사·GPU compute query에서 모두 없음/명령RC0을 확인했다.
원래 status/journal/source freeze/outer 전체로그와 별도새output의 CPU보고서6개를 회수해
remote hash/size113개 binding 및 회수파일 byte 검사를 통과했다. 증거는
`analysis_exports/g6_runtime_20261005/final_closed_controller/`이다.
보고서 첫 호출의 PYTHONPATH 누락(ModuleNotFoundError)은 실행 환경변수를 명시한 두 번째 CPU호출로
해결했다. GPU재실행·원래실행/원문/checkpoint 수정은 없으며 보고서는 종료/재채점 검증의 대체가 아니다.
FML 원문/checkpoint native SCP77824는 아직 진행 중이며 로컬 tensor/raw CPU검증 및 전체 통합gate는
아직 완료하지 않았다. 다운로드를 재시작하거나 부분 파일을 덮어쓰지 않는다.
그 검증 후 정확한 소유Pod삭제·API부재/저장소정리 확인→최종분석/PPT 반영을 진행한다.
새Pod/50K/900/G7/GitHub게시는 없다. 이전FL/FM/FV 검증자료와V8 baseline은 불변이다.

## 2026-10-05 13:43 UTC — 마지막 FML 학습·fresh reload 정상 종료

원래 journal에서 FML train EXITED0을13:43:34.465016UTC(8,005.5110초),
별도PID8025 reload EXITED0을13:43:40.891976UTC(6.3833초)에 확인했다.
이는 실제 단계 종료 증거이며 전체FML/네조건 완료나 로컬payload 검증 완료는 아니다.
같은447c0fb controller는13:43:40.959141UTC FML MMMU60 PID8156을 시작했다.
13:49:13UTC 소유Pod22u0uqi171v0cn API RUNNING/cost0.22USDh,
13:49:20UTC SSH에서1812/1813/8156 생존·GPU43%/10344MiB·4/60저장응답을 확인했다.
observer1312는13:53:30UTC까지 살아 있고 MMMU 종료 이벤트는 아직 없다.
다음 API/SSH 점검13:59:13UTC. 이후 외부60·전체controller/outer watchdog 종료,
모든 원문/체크포인트 회수·통합 로컬검증·정확한소유Pod삭제·최종분석/PPT가 남아 있다.
독립 읽기전용 gate 점검으로 기존report.py의447c0fb byte동일성(SHA64667fa5…)과
별도새output에서의 재사용 가능성을 확인했다. 단, report의 완료표시는 controller/outer
종료 또는 raw CPU재채점 증명을 대신하지 않는다. 아직 최종report를 생성하지 않았다.
원래네10K/4K평가/최초금융기한·예약·공개본/PPT/기존회수본은 불변이며 새유료작업은 없다.

## 2026-10-05 13:39 UTC — FML 9,664/10,000·정상 실행 관찰

13:39:03UTC 실제 API에서 소유Pod22u0uqi171v0cn RUNNING을 확인했다.
13:39:11UTC SSH에서 기존1812/1813/7418 생존·GPU100%/23290MiB,
FML 닫힌604/625 update·9,664/10,000소비·M/L 양쪽 실제gradient/update를 확인했다.
observer1312는13:40:01UTC에도 살아 있으며 학습 종료 이벤트는 아직 없다.
이는 verified wait이며 관찰 timeout을 이유로 실행을 재시작하거나 중단하지 않았다.
다음 API·SSH 점검13:49:03UTC. 남은학습→freshreload→MMMU60/외부60,
전체controller status/journal·outer watchdog 종료증거 회수→네조건 통합로컬검증→
정확한소유Pod삭제→최종분석/PPT 반영은 아직 남아 있다.
최종PPT 변경위치19/27/31/33/34/41/42를 독립 읽기전용으로 확인했으나,
최종 FML/전체회수/삭제 근거가 없어 원본42장과10개P1캡처는 변경하지 않았다.
최초 금융기한/예약·447c0fb·네10K·동일4K평가·기존FL/FM/FV 회수본은 불변이다.
새GPU/50K/900·실행중 패치·GitHub 게시를 시작하지 않았다.

## 2026-10-05 13:06 UTC — FML 7,232/10,000·실행 유지 체크포인트

13:06:22UTC 실제 소유Pod22u0uqi171v0cn는 RUNNING/cost0.22USDh였다.
13:06:25UTC SSH에서 기존1812/1813/7418 생존·GPU100%/23290MiB와
FML 닫힌452/625 update·7,232/10,000소비·M/L 두group 실제gradient/update를 확인했다.
observer1312는13:09:45UTC까지 살아 있으며 학습 종료 이벤트는 아직 없다.
이는 기존 실행의 verified wait이며 관찰 timeout으로 재시작/중단하지 않았다.
다음 API·SSH 점검13:16:22UTC. FML의 남은학습→freshreload→MMMU60/외부60,
전체controller status/journal·outerlog 회수→네조건 통합로컬검증→정확한소유Pod삭제→
최종분석/PPT 반영은 계속 남아 있다. 기존FL/FM/FV 검증회수본과 최초금융기한/예약,
447c0fb·네10K·동일4K평가를 유지한다. 새GPU/50K/900은 시작하지 않았다.

## 2026-10-05 12:35 UTC — FML 4,896/10,000·기존 실행 관찰 유지

12:35:16UTC 실제 소유Pod22u0uqi171v0cn API RUNNING/cost0.22USDh,
12:35:20UTC SSH에서 기존1812/1813/7418 생존·GPU99%/23290MiB,
FML 닫힌306/625 update·4,896/10,000소비와 M/L 두group gradient/update를 확인했다.
12:30:09UTC controller HEALTH_POLL도 같은PID 생존을 확인했다. observer1312를 재사용한
10분 관찰 주기는12:37:50UTC 종료 이벤트 없이 끝났으며 학습 자체는 중지하지 않았다.
다음 계정·SSH 확인12:45:16UTC. 이는 verified wait이며 차단/실험완료가 아니다.
FML 재로드·MMMU60·외부60 및 전체controller/outer 종료 증거 회수·통합검증은 아직 남아 있다.
완료 후 전체 결과 검증→정확한 소유Pod삭제→최종 분석/PPT 반영을 계속한다.
최초 금융기한/예약·447c0fb·네10K·동일4K평가와 기존FL/FM/FV 회수본은 불변이다.
새GPU/50K/900이나 실행 중 코드 수정은 하지 않았다.

## 2026-10-05 12:04 UTC — 마지막 FML 학습 생존 확인·회수 대기 체크포인트

12:04:01UTC 실제 RunPod API에서 소유Pod22u0uqi171v0cn RUNNING/cost0.22USDh,
12:04:05UTC SSH에서 기존1812/1813/7418 생존·GPU99%/20424MiB를 확인했다.
FML 닫힌 update158/625·consumed2528/10000이며 M/L 양쪽 실제 gradient/update가 기록됐다.
observer1312는 살아 있고 종료 이벤트는 아직 없다. 관찰 timeout은 작업 종료나 재시작 사유가 아니다.
다음 계정·SSH 점검12:14:01UTC. FL/FM/FV는 회수·CPU검증 완료지만 FML 학습/재로드/두 평가,
전체 status/journal 및 outer watchdog 종료 증거 회수·통합 로컬검증이 남아 있다.
그 검증 후에만 정확한 소유Pod삭제→최종 네 조건 분석/PPT 갱신을 진행한다.
원래447c0fb·네10K·4K평가·최초 금융기한/예약은 불변이며 새Pod/50K/900을 시작하지 않았다.

## 2026-10-05 11:36 UTC — FV full10K 로컬 검증 완료·마지막 FML 실제 학습

FV train/reload/MMMU/external EXITED0 및 두 input gate PASS 이후20개닫힌 파일과
33,487byte journal prefix를 회수했다. 정확한PodID/447c0fb/root·파일hash·10K/625/노출·
96 Vision LoRA FP32tensor/2,359,296 params·freshreload·동일평가입력·실제frozen scorer의
원문CPU재채점·후검사 PASS다. 기존 검증한FL processor/token 노출을 재사용했으며
base모델로드/이미지재처리는 없었다. frozen base 불변은 sampled checks이지 전체 인증은 아니다.

동일4K F036/60→FV21/60(35%,회복0/악화15). 외부QA4→6/12,TRACE7→6/24,
structure0/12·grounded0/12다. 독립Sol의산술/coverage/input/metadata 감사도 일치했다.
실제rule target2개는 F0/FV모두0/2이며 기존10/12·9/11은null/null일치라 mastery로 말하지 않는다.
FV structureF1 0.255590은11관측/1누락으로F0 0.201519의12관측과coverage를함께보고한다.
감사: `analysis_exports/g6_runtime_20261005/closed_full10k_FV/independent_metric_audit.md`.

전체 회수/체크포인트: `/home/jang2/mmdl-artifacts/recovered/g6-full-FV-20261005-22u/`.
Windows원문/집계/receipt/검증기byte복사: `analysis_exports/g6_runtime_20261005/closed_full10k_FV/`.
receiptSHA66b2dcb2aaea1073c19d39bb55d29ced16ea8de019828c3190cc82991d4eb449;
원문dir byte동일/검증기hash/Git제외 확인. 기존FV/FL/FM 응답/채점/공개본/PPT는 불변이다.

원래 controller는11:30:08.954813UTC FML train PID7418을 시작했다.
11:36:10UTC 실제1812/1813/7418 생존·GPU99%/22656MiB,
닫힌27update/432소비·최신position444와M/L두group실제gradient/update를 확인했다.
아직 FML학습/재로드/두평가 및 전체controller 종료가 남아 있어 Pod를삭제하지 않는다.
11:31:52UTC 실제API RUNNING/cost0.22USDh; 다음11:41:52UTC·observer1312 유지.
최초budget/deadline/reserve/원래네10K/4K평가 불변. 모든결과/종료증거 검증후
정확한소유Pod삭제→전체분석/최종PPT 반영을 계속한다. 새GPU/50K/900 자동시작 없음.

## 2026-10-05 11:25 UTC — FV MMMU60 완료·외부 평가 진행

원래 journal의 FV MMMU EXITED0(11:23:34.058210UTC,2571.5737초)와
mmmu_input_gate PASS(cb7ea84890a28ab755103f4dd76f89ddeb71ccdf34dec49a15b2335ca8240e5b)를 확인했다.
원격 summary는60/60·21정답(35.00%)·length15/stop45·system budget stop0이다.
이는 원격 완료 집계이며 아직 원문 CPU 재채점/로컬 회수 검증을 완료한 FV 결과는 아니다.
기존900/32K baseline과 직접 비교하지 않는다. FV 외부60은11:23:34UTC 새PID7208로 시작했고
11:24:58UTC 실제1812/1813/7208 생존·GPU70%/9244MiB·10/60을 확인했다. FML은 아직 대기다.

외부 input gate PASS까지 닫히면 기존회수기로 FV 체크포인트/원문/로그/hash를 회수·검증한다.
수신대상 `/home/jang2/mmdl-artifacts/recovered/g6-full-FV-20261005-22u`는 아직 없고,
Linux783GiB/Windows217GiB 가용을 확인했다. Observer1312·다음API11:31:07UTC 유지.
원래 실행·기한·예약·공개본/PPT/평가조건은 불변이며, 모든결과 로컬검증 전에 Pod삭제하지 않는다.

## 2026-10-05 11:11 UTC — 복합 병목 관련 문헌 조사·기존 FV 평가 유지

sc-research로 논문/저자 자료를 확인하고 내부
`claudedocs/research_visual_structure_reasoning_training_20261005.md`에 기록했다.
MMFineReason의 Qwen3-VL-4B-Instruct SFT·MMMU-val67.4→69.3은 저자 보고값이며,
우리 revision/P0/V8/3090 재현 결과가 아니다. G-LLaVA/Geo170K, LLaVA-CoT,
DocVAL 및 간결한 grounded CoT 연구의 데이터·검증·평가 차이와 후속 후보를 구분했다.
LLaVA-CoT 최신 전체평균의 retracing 표기 충돌은 SFT-only 효과로 인용하지 않았다.
보고서는 Git 제외 확인·로컬 근거 파일 존재·diff whitespace 검사 PASS다.
새 데이터 취득/학습/API 호출/GitHub/PPT 변경은 하지 않았다.

11:10:51UTC 실제 API는 소유Pod22u0uqi171v0cn RUNNING/cost0.22USDh,
11:10:55UTC SSH는 기존1812/1813/6859 생존·GPU68%/10430MiB와 FV MMMU43/60을 확인했다.
FV 외부60/FML은 아직 완료되지 않았다. 다음 계정/SSH 점검11:20:51UTC,
journal observer session1312 유지. 원래447c0fb/네10K/4K평가/기한/예약은 불변이다.
모든 닫힌 결과 회수·검증→정확한 소유Pod삭제→최종 분석/PPT 반영을 계속한다.

## 2026-10-05 10:41 UTC — FV 학습·새 프로세스 재로드 완료, MMMU60 평가 진행

원래 controller journal에서 FV train EXITED0(10:40:38.529582UTC,
8011.4483초), reload EXITED0(10:40:42.414295UTC, 새PID6728)를 확인했다.
FV 소비 원장의 마지막 position9999는10,000개 모두 소비했음을 보여 준다.
10:40:42.485427UTC FV MMMU60 평가가 실제PID6859로 시작했다. 외부60과
FML은 아직 완료되지 않았으므로 FV 점수·네 arm 전체 결과는 확정하지 않는다.
10:41:08UTC RunPod API는 소유Pod22u0uqi171v0cn RUNNING/cost0.22USDh,
10:41:14UTC SSH는 기존 부모1812/1813 생존·GPU68%/9262MiB를 확인했다.
다음 계정 점검은10:51:08UTC이며 journal observer session1312를 재사용한다.

PPT42장/실제P1캡처10개의 최종 검증 receipt와 원래18장 보존은 기존09:29UTC
기록 그대로다. 이번에는 슬라이드를 반복 수정하지 않았다. 남은 실험 종료 후
회수·CPU 재채점·전체 종료 증거 검증→정확한 소유Pod삭제→최종 분석/PPT 반영을
계속한다. 원래447c0fb 코드·4K 평가·네10K조건·과금 시작/기한/예약은 불변이며
G7/50K/새900/새Pod를 자동 시작하지 않는다.

## 2026-10-05 09:35 UTC — 원래 네 arm 실행 유지·최종 회수 범위 확인

09:38:56UTC `ps`에서 실제1812/1813/6111 생존을 다시 확인했다. 읽기 전용SSH의
`tail -F journal.jsonl`을 unified exec session1312로 유지한다. 최근09:37:07UTC
HEALTH_POLL은 학습PID6111 생존·GPU100%다. 후속 턴은 이 session을 기다려 종료이벤트를 받는다.
observer가 끊겨도 GPU 작업 종료로 단정하거나 학습을 재시작하지 않는다. 직접PID/원장을 재확인한다.

09:34:43UTC RunPod 실제 API: Pod22u0uqi171v0cn 상태RUNNING, cost0.22USD/h,
container disk30GB, /workspace persistent160GB이며 network mount는 반환되지 않았다.
09:34:48UTC SSH: GPU100%/18264MiB, FV 마지막 기록position5093; 09:32:54UTC 원래 부모
1812/1813과 학습PID6111은 모두 생존했고 FV closed consumed4945였다. 공동FML은 아직 미시작이다.
다음API/SSH 점검09:44:43UTC. FL/FM 부분 검증만 완료됐으며 네 arm 전체 완료는 아니다.

읽기 전용으로 controller 루트 파일 목록과 run_experiment 종료 로직을 확인했다. 실행 중
status.json이 없는 것은 예상된 상태이며, controller 종료 시 작성되는 파일이다. 결과 유실이나
실행 중단의 증거가 아니다. 최종에는 닫힌 status.json·전체journal.jsonl 및
/workspace/g6-control/g6_3090.log까지 회수해 마지막controller 이벤트·네10K/625·종료상태를 확인한다.
각 arm receipt는 external_input_gate까지만 보존하므로 전체 종료 증거를 대신하지 않는다.
환경/고정source는 기존 검증SHA를 재사용한다. 로컬build_report를 통과시키려고 원래summary의
/workspace 절대checkpoint 경로를 바꾸지 않는다. 기존paired_counts/external_groups와 명시적인
remote→local 결합을 사용한다. 독립Sol이 이 gate를 읽기 전용으로 검토했다.
/home에784GiB·Windows볼륨에217GiB가 남아 회수 공간이 충분하다. 새Pod/추가GPU작업/
deadline 재설정/실행 중447c0fb 수정은 없으며, 최종 전체 로컬검증 전 Pod를 삭제하지 않는다.

## 2026-10-05 09:29 UTC — PPT 근거 보강 검증·외부 규칙 지표의 분모 정정

PPT42장 중31/33/42만 갱신했다. 원래1–18 slide/notes/media byte와 나머지 슬라이드는
보존했고 P1 실제캡처10개의 내장SHA/원본HTML 불변, native 렌더3장 시각검토,
overflow0, native42/notes/citation 검사와 byte 검사 PASS다. 현재 PPT SHA는
`57ff8e303c18dba1716f9034ec3a032625c542a6f616d9356c2b071d51161f96`이다.
FM 로컬검증·27/60을 상태에 반영했고 완료되지 않은 FV/FML은 진행/대기로 남겼다.
검증: `analysis_exports/ppt_updates_20261005/verification_final.json`;
이전42장 전체백업: `analysis_exports/ppt_updates_20261005/review_corrections/original_42_slides.pptx`.

독립Sol은 frozen evaluator로 grounded36개(F0/FL/FM)를 CPU 재현했다. 실제 rules target은
12개 중2개이고 F0/FL/FM 모두 rule-present exact0/2다. 과거 rule-exact10/12→6/12(FL),
10/12(FM)는 규칙이 없는 target과 prediction의 null/null 일치를 포함한다. 이를 규칙 학습
성공률로 말하지 않으며 원래 frozen 점수/원문/채점기는 바꾸지 않는다. grounded 최종답의
회복 중 FL2/FM2개는 동일 수치의 문자열→숫자형만 바꿔도 회수되는 사례다. 최종답 지표의
상승과 관계/상태/규칙 학습 성공을 구분한다. MMMU answer-only JSON은 세 조건 모두0개;
단독 LETTER.보기 응답은 F0 2→FL51/FM25개로, FL의 짧은 label/보기 형식 적응이 관측됐다.
근거13파일 SHA 불변과 원장 재현을 확인했으며 GPU/네트워크/새추론은 사용하지 않았다.
증거: `analysis_exports/g6_runtime_20261005/partial_FL_FM_metric_audit.json`.

09:25:11UTC SSH에서 RTX3090 GPU100%/18264MiB·FV 학습을 확인했다. 최근 controller
HEALTH_POLL09:17:07UTC는 PID6111 생존·GPU100%다. 다음 계정·SSH 점검은09:35UTC다.
원래447c0fb 실행·네10K조건·deadline/reserve 불변. FV/FML 회수·검증 전 Pod삭제는 하지 않는다.
모든 결과 회수·검증→정확한소유Pod삭제→전체분석을 계속한다. G7/50K/최종900 자동실행 없음.

## 2026-10-05 09:14 UTC — FM full10K 회수·원문 재채점 검증 완료

원격닫힌15개FM파일+3개freeze와25,240byte journal prefix를회수했다. `/proc/1`의실제PodID,
실제Git447c0fb/실제root, 파일크기/SHA, 종료0·정확한train/reload/eval인자·다른4PID/단계순서,
10K/625·동일records/order/input/target/visual 노출,24개Merger FP32tensor의shape/finite/hash,
freshreload 및 두60개의실제입력·seed/token·고정채점기를사용한CPU원문재채점이PASS다.
독립Sol이회수guard를검토했고Ruff/3파일boundedmypy/잘못된arm거부검사도PASS다.
누락된snapshot audit.py를현재고정SHA에직접결합하고, 현재G7 evaluator 대신실제447c0fb frozen
evaluate.py를재사용하도록회수도구만수정했다. 실제GPU실행/응답/파서는변경하지않았다.

MMMU60 동일4K F0 36/60→FM27/60(45.00%), +2/−11이다. 외부development유형별로
QA4→7/12, TRACE-QA7→11/24, whole-structure exact0→0/12, grounded0→5/12다.
grounded state-exact0→0/12, rule-exact10→10/12이므로최종답성공을상태/규칙향상으로말하지않는다.
StructureF1은0.19548(6관측/6미관측); F0 0.20152(12관측/0미관측)와누락률을함께보고한다.
외부혼합지표23/60은MMMU점수가아니며공식62%/32K900과직접비교하지않는다.
아직FV/FML이진행/대기중이므로모듈선택이나대규모본학습확장을시작하지않는다.

원본체크포인트/전체회수: `/home/jang2/mmdl-artifacts/recovered/g6-full-FM-20261005-22u/`.
Windows원문/집계/receipt: `analysis_exports/g6_runtime_20261005/closed_full10k_FM/`.
회수도구: `analysis_exports/tools/recover_g6_closed_arm_20261005.py`; 비공개이며MMDL/GitHub미게시다.
최종모든결과회수·검증후소유Pod삭제를유지한다. 현재발표PPT는회수전17:49KST상태스냅샷이며,
남은arm종료후새결과표를반영한다. 기존1–18/고정baseline/P1원문은그대로보존한다.

## 2026-10-05 09:03 UTC — PPT 근거 화면 보강 완료·원래 네 arm 유지

`qwen3_mmmu_problem_improvement.pptx`는42장이다. 원래1–18장 slide/notes/media byte는
모두보존했다.19–34에는 실제baseline/조건별진단/Knowledge Oracle/학습/한계를,35–42에는
P1실제전체/오답 화면 및 CS16·Accounting9·Manage20·Music3의입력이미지/응답말미10캡처,
Qwen/CLEVR/LoRA 1차출처·HW실측·비용기한·마일스톤과평가요소연결을추가했다.
원장/HTML/모델입력/파서/558점은불변이며캡처만내부발표자료에사용한다.GitHub미게시다.
긴페이지스크롤의빈캡처는사용하지않고원문DOM발췌로재촬영했고Sol독립PNG확인/부모native8장/
overflow0/42장·원래18 signature/원래slide·notes·media byte/임베드10 SHA검사를PASS했다.
검증: `analysis_exports/ppt_updates_20261005/verification_final.json`, PPT SHA
`91314b3d5ac3e007e054cb4e6ee663eb1f8baf2ffd571b1b6c2b54b2ebc8e679`.
33장후속선택은외부dev만사용하고MMMU유지위험별도보고로정정했으며실제사전등록경로를수정했다.

09:00:06UTC 소유3090 Pod22u0uqi171v0cn는RUNNING·FV2456/10000·GPU95%/15402MiB다.
FL10K/625와MMMU/외부60개원문은이미로컬검증했고MMMU는동일4K F036→FL22/60이다.
FMtrain/reload/MMMU/external EXITED0·08:27:07UTC inputgate통과했으며현재로컬회수검증을진행한다.
FV학습/FML대기중이므로Pod를아직삭제하지않는다.기존deadline/reserve/4arm/protocol447c0fb불변이다.
최신사용자지시에따라모두종료후모든결과다운로드·원격SHA/실제tensor/CPU재채점검증→정확한
소유Pod삭제→분석한다.G7/50K/8K전환/최종900을자동시작하지않는다.10분상태점검을유지한다.

## 2026-10-05 05:08 UTC — 실제 회로 소비 원장 결합 완료·본학습 계속

05:06:19UTC 계정조회에서소유3090는RUNNING/0.22USDh였고,05:06:22UTC SSH에서FL6733/10000,
닫힌update420/소비6720·GPU99%/20800MiB와부모/학습생존을확인했다.다음조회05:16:19UTC다.
실행447c0fb·원래네10K조건·2배gate·최초과금/기한/회수예약은불변이고FT점수는아직없다.

실제후속training_bundle 소비검사는369.382초에1089파일/네231행/dev120/MMMU60을PASS했다.
원장 controller_consumer_v5_input_bound.json은정확한호출경로·21metadata SHA·1089파일원장/
digest·47source before/after freeze·v4 SHA까지결합한다.독립Sol은digest/metadata/47source를
확인했고이미지전체재검사/GPU/모델로드는하지않았다.실패v1–v3와원장부족v4는그대로보존한다.
근거 analysis_exports/g7_circuit_pool100_20261005/; 메모리fit/회로학습/fullG7/전이완료는아니다.

full10K 외부결과를보기전에후속모듈선택규칙을비공개로사전고정했다.
primary는qa12와trace_qa24의유형별정확도동일가중치(순위2×qa정답+trace정답),보조는기존
structure/grounded 성공flag다.후자의state/rule 정확도는별도보고하며엄격구조성공으로혼동하지않는다.
모든네full10K/재로드/동일입력/CPU 재채점/로컬회수검증전선택하지않는다.MMMU·봉인·실제전이는
선택입력이아니다.F0 rank15/48 이하이면개선승자가없으며bounded probe와효과입증을구분한다.
50K 자동확장/유료추가실행은없다.근거 analysis_exports/g6_runtime_20261005/module_selection_preregistration.json.
현재source/후속준비만MMDL_test에반영하고live checkout·교수님MMDL/V8는변경하지않는다.

## 2026-10-05 04:34 UTC — FL 본학습 진행·회로100root 후속 입력 실제 준비

소유3090 Pod22u0uqi171v0cn는RUNNING/0.22USDh다.04:34:23UTC FL 본학습4284/10000,
닫힌update267/소비4272·GPU100%/20800MiB·allocated peak12822514176bytes를확인했다.
outer1812/inner1813/train3832와600초HEALTH_POLL이생존하며본학습완료/FT점수는아직없다.
다음계정점검04:44:20UTC다.최초과금·금융기한·회수예약·4arm/full10K/2배gate는불변이다.

기존circuit_bridge만재사용하여 별도비공개100root(97family) 풀을고정했다.
train77root/75family, development10root/10family, sealed13root/12family다.봉인root는
학습/development선택/모델평가에쓰지않고전체CPU전처리·준비검증에만포함했다.
두독립exact solver·canonical/render 재현은100root 모두PASS다.독립Sol이실제6개PNG의
값/연결/극성/스위치/재배치를검토했고메인도2개를열람했다.이는AI표본검토이지사람의
독립승인/전체픽셀의미인증은아니다.한가지ideal DC topology이고일반전자공학을대표하지않는다.
실제Pillowfont의출처/권리/hash를기록했으며OPEN/CLOSED표기와값/단위의shortcut가능성도남겼다.

현재audit와pinned Qwen processor로1200행을실제CPU전처리했다(각A0–A3 300행,
maxsequence3194/assistant231/child sequence4096/episode3145728).기저가중치/GPU는로드하지않았다.
일반audit.approve 경로로각231행/77root/75family를새승인했고quarantine0이다.
원본pool의training_allowed=false와과거receipt는변경하지않았다.제외index는검증된
100파일/16740행의기존fingerprint를경로투영한것이며원본Parquet를새로감사했다고하지않는다.
같은10development root×4조건×3view=120행을gold-blind로별도고정했다.
근거 analysis_exports/g7_circuit_pool100_20261005/{actual_processor_summary,preparation_receipt}.json.

기존Graph controller에Circuit routing/실제행수·root수·typed score/새이름만추가했다.
freshF0 네조건·고정모듈·source freeze/승인·family분리·금융gate는유지한다.
독립검토에서Circuit gold가generic qa로바뀌면V8채점으로잘못연결될수있는경로를발견했고,
공유circuit_metric_gate를학습시작전과독립결과검증에모두적용했다.최종191회귀검사
(798.773초)/Ruff/mypy3/게시안전3검사/게시checkout표적2검사는모두PASS다.
초기새테스트의실패는immutable artifact writer로fixture를덮어쓰려던오류였으며fixture쓰기만고쳤다.
운영writer/기존증거/검사는약화하지않았다.소스편집과겹친소비검사는최종source인증으로쓰지않는다.
후속source-frozen 소비검사는입력/hash/freeze까지통과했으나검증명령의UTC함수오용으로
receipt쓰기가실패했다.그실패도보존하고표준datetime을쓴최종receipt검사를진행중이다.
GPU메모리적합성/모듈선택/회로학습/fullG7/비합성전이는미검증이며G6 외부결과를기다린다.
현재GPU447c0fb checkout과교수님MMDL/V8 558/900·저장원문은불변이다.

## 2026-10-05 03:53 UTC — 네영역 계측·로컬 tensor 검증 완료, original10K 본학습 진입

소유3090 Pod22u0uqi171v0cn의 FL/FM/FV/FML 360행 학습은모두실제360소비/23updates/
EPOCH_COMPLETE와fresh-process reload exit0으로닫혔다.학습metadata 시간은각316.761,
249.491,310.112,321.722초다.동일한순서·348194입력token/68274target/270000visual/
360image 노출을확인했다.이는memory/update/save/reload 계측이며본학습성적이아니다.

03:37:27.415UTC 기존2배forecast gate는필요79942.903초(22.206시간),
잔여82780.886초(22.995시간)로통과했다.본학습량을축소하거나margin을바꾸지않았다.
03:37:27.461UTC original10000행 FL 본학습PID3832가freshF0에서시작됐고,
03:50:05UTC940소비/update58(닫힌update소비928),GPU98%/12590MiB를확인했다.
allocated peak12384083456bytes/reserved peak24584912896bytes이며OOM은관측하지않았다.
현재outer1812/inner1813와600초HEALTH_POLL도생존한다.다른arm본학습/FT평가는미완료다.
이예측은G6 네종본학습+60문항MMMU/외부평가뿐이며 G7/전이/최종900시간을포함하지않는다.

닫힌네계측의27파일과8로그를기저모델/캐시없이로컬로회수했다.
shared verify_arm_training을재사용하고원격file SHA·processor receipt 연결·per-ID입력/target/
visual grid합계·순서/노출/실제업데이트 및 모든1152개trainable tensor의key/shape/dtype/
numel/finite/stateSHA를CPU에서검증했다.전체원격journal이아직append중이므로11404bytes의
닫힌stage prefix만독립적으로원격SHA와대조했다.원문경로를변경하거나terminal상태를만들지않았다.
reload명령의module/pinned base/checkpoint 및train종료→reload시작순서도확인했다.
처음독립Sol 검토의두evidence gap은보완했고실제최종검증PASS다.
four_arm_local_receipt_v2.json과smoke_journal_prefix_remote_receipt.json이근거다.
원래processor producer bba0ed1b…의증거를재사용했으며새전처리/전체frozen-weight인증은아니다.

비공개회수도구 analysis_exports/tools/verify_g6_closed_smokes.py는Git 제외이며
모델·학습code·채점·protocol·live447c0fb checkout을변경하지않는다.교수님MMDL/V8 558/900도불변이다.
게시된실험source68216dea…와liveexecution447c0fb를구분한다.원예산8.42/총요율상한0.30/h/
회수예약1USD/최초과금01:55:08.301UTC/종료10/06 02:37:08.301UTC는유지한다.
03:50:02UTC API는RUNNING/0.22USDh이며다음계정점검04:00:02UTC다.
마감시프로세스종료만으로과금이멎지않으므로회수·검증후정확한Pod삭제를별도로수행한다.
최종900/32768평가는아직시작하지않았고60/4096진단은기존전체baseline과직접비교하지않는다.

## 2026-10-05 03:18 UTC — 새3090 F0 기준선120개 회수·첫 학습 계측 진입

소유Pod22u0uqi171v0cn는 RUNNING이다. 새F0 MMMU60은36정답/60(60.00%),
length12/stop48, evaluator2581.072초로 완료했다(03:06:37UTC 단계exit0).
외부60도11/60·length2/stop58·607.039초로 완료했다(03:16:48UTC 단계exit0).
외부60은 QA/구조/grounded/TRACE를 섞은 지표이며 MMMU 정확도가 아니다.
두120개 원문·summary와 실제 실행evaluate.py를 로컬로 회수하여 원격SHA,
60×2 ID/순서·고정cohort/protocol/원source·실제seed/token/이미지관측 및 CPU 점수 재현을 검증했다.
비공개 경로 analysis_exports/g6_runtime_20261005/closed_stages/의 각local_receipt.json을 따른다.
이4096토큰 진단 기준선은 기존V8 900/32768의558/900을 대체하지 않는다.

03:16:48UTC 첫FL smoke 학습PID2476이 시작됐다.03:18:20UTC GPU51%·19202MiB,
controller1813/outer1812 생존을 확인했고03:19경 소비150/360·실제update9개를 관측했다.
이는 네학습영역의360개 학습/재로딩 시간·메모리 계측이며 full10K 본학습 완료가 아니다.
과금시작·금융기한/회수예약1USD·기본2배 full10K4종 gate는 불변이다.
다음 계정점검은03:28:16UTC이며 완료 결과를 검증 후 정확한Pod 삭제 원칙을 유지한다.

후속 회로48행은 실제 pinned processor CPU 전처리를 통과했다(maxsequence3192/assistant230).
그러나 실제4root 후보는 train3/sealed1/development0여서 새개발 cohort 소비 검사가
미승인학습/봉인root 대체 없이 BLOCKED로 종료했고 출력 폴더를 만들지 않았다.
원본 후보/hash/48개 processor 증거는 보존한다.
analysis_exports/g7_circuit_processor_20261005/{summary,cohort_readiness}.json이 근거다.

기존 bridge_evaluation/evaluate에 bounded Circuit v2 경로만 추가했다.
동일 개발root×A0–A3×3views, gold-blind whitelist, 독립solver/render 재현,
current producer/child protocol/typed proof·train approval·root/family 분리,
checkpoint의완전학습manifest/records/exposure 결합과 정직한 metric metadata를 검사한다.
output이 입력pool/approval 내부면 거부한다. 독립Sol의두 guard지적은 수정했고 재검토 PASS다.
새6 CPU 계약tests(49.538초)·최종source 전체188tests(304.467초)·Ruff·mypy3·게시안전3검사 PASS다.
테스트용 구성receipt는 실제processor/권리/오염승인이나 GPU 학습 증거가 아니다.
새builder로 별도Graph24 cohort를 다시 고정하고 실제 training_bundle440파일 소비도 PASS다.
기존 approval/cohort는 덮어쓰지 않았다. consumer_after_circuit_integration.json에 근거를 남겼다.

새코드는 본학습/G7/비합성 전이 완료가 아니며 현재Pod의447c0fb 실행을 hotpatch하지 않는다.
교수님MMDL·기존V8·모델/BF16/데이터/본학습량/출력 조건은 불변이다.
초기 새테스트의두실패는 immutable audit.dump로fixture를 덮어쓰려던 테스트오류였다.
변조검사만명시적fixture쓰기로나누어수정했으며 runtime writer/기존receipt는변경하지않았다.
최종검사에서 source/guard를약화하거나테스트를생략하지않았다.

## 2026-10-05 02:48 UTC — GPU 평가 지속·현재 source 후속 입력 CPU 승인 완료

10분 계정 점검에서 소유3090 Pod22u0uqi171v0cn는 RUNNING이었다.
02:48:46UTC evaluator1878/inner1813/outer1812 생존, F0 MMMU33/60·GPU73%·10096MiB를 확인했다.
600초 HEALTH_POLL도 진행 중이며 다음 계정 점검은02:58:46UTC다. 본학습/최종900 평가는 아직 아니다.

별도 로컬 CPU에서 현재 audit.py5fb37650…와 pinned Qwen processor로384개 bridge 후보를
실제로 전처리했다. A0–A3 각84개(같은28 train roots)만 새 승인했고 development24는 분리했다.
현재 run_bridge_experiment.training_bundle 소비 검사가440개 파일/hash·실제 이미지/역할·
cohort/승인 결합 및 분리를 통과했다. 독립 비합성 전이 도구의13 CPU tests도 PASS다.
CUDA는 숨겨서 실행했고 모델 가중치 로드·학습·추론은 하지 않았다.

처음 승인 시 기존 제외 index의 coverage snapshot 경로를 로컬에서 resolve하면 원래 서버의
blob 경로와 달라져 fail-closed 검사가 거부했다. 원본SHA f062a53a…를 고정하고100개 파일의
SHA/행수/저장소 namespace를 일대일 대조한 뒤, 별도 비공개 index의 coverage 경로만
기존 검증 blob 경로로 투영했다. source_files/행수/모든 fingerprint/권한/한계는 불변이다.
원래 coverage와 전체 대응표를 provenance에 보존했고 잘못된SHA/행수 반례2개를 거부했다.
독립Sol 검토 PASS이며 원본 index/18개 과거 proof는 byte 불변이다. 원본Parquet를 재검증하거나
benchmark fingerprints를 다시 만들었다고 주장하지 않는다. 현재 runtime guard는 변경하지 않았다.

비공개 산출물은 analysis_exports/g7_cpu_refresh_20261005/에만 보관한다.
summary.json·consumer_receipt.json·새 proof/개발 cohort·부모SHA 결합 index를 포함한다.
이 결과는 bounded synthetic Graph/CS 입력 준비이며 full professional G7/독립 비합성 전이의
실험 완료가 아니다. 현재의84행 GPU 제어기를 큰 전문분야 실험 완료로 표시하지 않는다.
후속 비합성 ChartQA_Human 전이 준비에는 실제 완료된 checkpoint-bound 학습 증거가 필요하다.

현재 Pod의447c0fb 고정 코드·G6 입력/4종/full10K/기본2배 예산 여유·과금 시작/금융 한도는 불변이다.
기존 V8 558/900와 교수님MMDL도 불변이다. 새 입력/대형 결과는 GitHub로 게시하지 않는다.
진행 근거 analysis_exports/g6_cloud_deployment_20261005.json; 전체 목표는 계속 미완료다.

## 2026-10-05 02:24 UTC — 새 clean GitHub 버전 검증·3090 GPU 기준 평가 진입

역사적 processor fixture를 현재 승인이라고 주장하던 두 테스트만 수정했다.
수정본447c0fb6056d2e18618dd22f21ef80197cb57103는 MMDL_test feature branch에 정상 게시됐다.
unmocked 현재 source는 과거 producer를 거부하고, 양성 테스트는 명시적 역사 fixture의
producer hash만 고정한다. 다른 파일 hash는 실제로 계산하며 receipt byte 변조를 거부한다.
runtime/승인기/원본 receipt/학습 설정은 변경하지 않았다. 독립Sol 검토 PASS다.
로컬 관련12 tests120.960초·전체182tests399.326초·Ruff/mypy2·게시 안전3tests PASS.
새 clean checkout /workspace/MMDL_test_g6_verified에서182tests103.130초·pip check·
GPU/BF16·storage doctor 및 지정 Qwen13파일 hash를 확인했다. setup exit0/130.951초다.

소유3090 Pod22u0uqi171v0cn에서02:23:32UTC baseline_F0_mmmu가 시작됐다.
02:24:03UTC 실제GPU66%·9260MiB를 확인했다. outer watchdog1812/innercontroller1813/
첫 evaluator1878, 로그 /workspace/g6-control/g6_3090.log다.
결과는 /workspace/nvw-run-verified/runs/g6_3090_resume_20261005에 따로 저장한다.
같은3090에서 새F0/네360개 smoke를 계측한 뒤 원래full10K4종·기본2배 예산 gate를 적용한다.
baseline은60/4096 진단 protocol이며 과거V8 900/32768 성적558/900을 대체하지 않는다.
평가60×2의 실제 모든 이미지/역할hash·고유ID·goldID일치도 별도CPU검사로 확인했다.
본학습 완료나900개 재평가 성공은 아직 아니다. final900/32K범위 질문은 미응답이다.

과금시작01:55:08.301UTC·잔액8.42·총요율상한0.30/h·회수예약1USD는 변경하지 않는다.
inner종료10/06 02:37:08.301UTC, outer종료02:39:08.301UTC다.
watchdog는 프로세스만 중단하므로 결과회수/로컬검증 후 정확한Pod 삭제가 별도로 필요하다.
기존 실패setup.log도 로컬 analysis_exports/g6_runtime_20261005/setup/initial_failed_setup.log에 보존했다.
증거 analysis_exports/g6_cloud_deployment_20261005.json. 교수님 MMDL/기존V8는 불변이다.

## 2026-10-05 02:07 UTC — 새3090 실측·설치 테스트 실패·비공개 복원 완료

기존4종/full10K/paired60를 유지하는 예산 내 적합성 검증은 재승인 범위다.
추가 범위 질문에 답이 없다는 이유만으로 가능한 안전한 진행을 중단하지 않는다.
01:55:08.301UTC 생성한 소유Pod22u0uqi171v0cn는 RUNNING이다.
실제RTX3090 24576MiB/driver580.65.06/Python3.12.3 및 persistent workspace를 확인했다.
GPU0.22USD/h·container30GB/workspace160GB·Network Volume 없음,
보수적 총요율0.30USD/h·회수예약1USD·금융종료10/06 02:39:08.301UTC다.
SSH의 누락된 Pod ID는 PID1 실체와 대조한 뒤 설정했다. 검사를 우회하거나 code를 패치하지 않았다.
GitHub clean clone은38b3d230176a018062b74ddfb228937243e7ebda다.
승인10K/smoke/cohort와 기존 닫힌 결과의 비공개 SSH 전송은 모두 exit0으로 끝났다.

패키지 설치·pip check 뒤182개 CPU tests(106.222초) 중2개가 실패했고 setup exit1이다.
두 실패는 저장 bridge processor의 producer SHA bba0ed1b…와 현재 audit.py5fb37650…의 차이다.
실제 GPU 학습 장애가 아니라 현재 source를 요구하는 guard와 역사적 fixture의 충돌이며,
검사를 약화하거나 receipt를 재작성하지 않는다. 테스트에서 현재 source 거부를 검증하고
역사적 fixture의 양성 경로는 명시적 합성 producer fixture로 분리해 검증한다.
정상 GitHub 게시/새 clean checkout/전체 검사를 통과한 뒤 모델·GPU gate와 본학습을 이어간다.
GPU 학습/추론은 아직 미시작이고 기존V8/교수님MMDL는 불변이다.
상태 근거 analysis_exports/g6_cloud_deployment_20261005.json,
원격 /workspace/g6-control/setup.log. 매600초 상태 확인·회수/검증 후 소유Pod 삭제를 유지한다.

## 2026-10-05 01:45 UTC — 10K 원본 전처리 증거·복원 입력 소비 검사 완료

누락으로 보였던 원본 processor receipt를 pilot_raw에서 찾았고 SHA256
732fb3eacbf40bab534395ed5d6ed0cd9e2f130d8bc292af18c5f088923961f3를 확인했다.
최종 row ID와 각 receipt hash로9951+25+24=10000개를 모두 연결해 PASS·길이·이미지 수·
tensor hash 필드를 확인했다. 실제 최대 input+assistant 길이는1643token/assistant867,
최대 처리 patch cell4096이다. smoke는1738token/3000patch cell이므로
짧은 전체 길이만으로24GB GPU 적합성을 입증하지는 않는다. 기존 receipt는 불변이다.

native f3 cwd·실제 module import 경로를 확인한 CPU 호출에서 approved_records 및
validate_training_input_contract가10000행·9773고유 이미지 SHA·공통 입력 계약을 통과했다.
현 학습 source와 byte 일치·smoke root 중첩0·full10K/무축소를 확인했다(exit0).
이전 common10k 폴더만 조사했을 때의 receipt 부재 한계는 해소됐으며
증거는 analysis_exports/g6_closed_results_20261004/processor_shape_recheck_20261005.json이다.
새 processor/GPU/학습/평가 실행이 아니며 paid 범위 질문은 아직 미응답이다.
계정 Pod0개를 재확인했고 새 대여·GitHub 변경·V8 결과 변경은 하지 않았다.

## 2026-10-05 01:36 UTC — 본학습·MMMU 평가 재승인, 실행 범위 확인 중

사용자가 본학습 후 MMMU 평가를 재승인했고 현재 잔액8.42USD를 보고했다.
목표는 다시 active이며 이전 BLOCKED 기록은 당시 상태로 보존한다. 새 Pod는 아직 생성하지 않았다.
실시간 계정 조회는 Pod0개다. CUDA>=12.8·GPU1개 카탈로그에서 A40 SECURE는
0.49USD/h·HIGH, RTX3090 COMMUNITY는0.22USD/h·LOW이며 A40/4090/A5000/A6000
COMMUNITY는NONE이다. 가격은 GPU 요금이며 storage/설치/복원 비용과 구분한다.

기존 full10K 네 조건·60문항 paired 평가의 재사용 forecast는83,478.931초이며,
기본2배 학습 여유를 유지한다. 새 setup/복원 및900문항/32K 평가·G7/독립 transfer는
이 수치에 포함되지 않는다. A40 요금에서 현 잔액으로 전 범위를 충족한다고 주장하지 않는다.
24GB 후보는 기존 smoke의 최대 Torch reserved21.914GiB·최장1738token 근거만 있어
4096token 본학습 전체 메모리 적합성과 속도가 검증된 것이 아니다.

사용자에게 MMMU 평가 범위(900 validation/기존60)·본학습 영역(기존4종/지정1종)을
질문했다. 학습량·정밀도·생성 조건·안전 여유를 임의로 바꾸거나 유료 실행을 시작하지 않는다.
기존 F0/smoke/승인10K 입력은 재사용하되 과거 receipt와 V8 558/900을 변경하지 않는다.
MMDL_test의 clean 로컬·원격 feature HEAD38b3d230176a018062b74ddfb228937243e7ebda
일치를 재확인했다. 새 GPU 실행·본학습 완료·새 점수는 아직 없다.

## 2026-10-04 17:49 UTC — 목표 BLOCKED: 추가 유료 실행의 예산/시간 승인 대기

동일한 추가 예산/최대 실행시간 승인 차단을 세 연속 목표 turn에서 확인했고
실제 목표 상태를 `BLOCKED`로 변경했다. 계정 Pod 목록0개·이전 Pod 회수/삭제 완료와
full10K 미시작 상태를 재확인했다. 비용 없는 재개 호환성/회귀 검증은 끝났다.
다음 실질 단계는 추가 GPU가 필요한 full10K 네 조건의 학습·재로딩·paired evaluation이다.
그 결과 없이는 G7의 근거 있는 모듈 선택과 학습된 모델의 독립 전이 평가도 진행할 수 없다.
과거 작은 실험이나360개 smoke로 전체 완료를 대체하지 않는다.
추가 예산/시간 확인 후 재개하며 자동 목표 계속 메시지는 새 지출 승인이 아니다.
범위는 G6 full10K→G7/전문 ablation→독립 비합성 transfer 그대로이고 완료로 표시하지 않는다.
감사 근거 `analysis_exports/g6_closed_results_20261004/blocking_audit.json`.

## 2026-10-04 17:46 UTC — 비용 없는 재개 호환성 검사 완료

기존 `--baseline-mmmu`/`--baseline-external`/`--reuse-smoke-controller`를 검토했다.
저장 F0 두60개의 실제 evaluation gate와 별도 evaluator source hash 검사 PASS,
f3/상위/게시 checkout의 학습·추론·파서·프롬프트·protocol 관련15개 파일 byte 일치 PASS다.
재사용/불완전 checkpoint 거부/예산/금융 한도 관련 기존4 CPU tests도 PASS(0.041초)다.
이는 원격 재개 성공/속도 측정이나 본학습 완료가 아니다.

현재 실제 원격 branch와 clean MMDL_test checkout HEAD는 `38b3d230176a018062b74ddfb228937243e7ebda`다.
현재 audit.py와 f3의 producer hash는 다르다. 독립 검토에서 변경 함수는
contamination_candidates/approve로 확인됐고 소비·재사용 경로의 조건은 동일하다.
기존 승인 자료는 불변으로 소비한다. 현 코드의 재승인/완성 경로에 과거 receipt를 넣어
producer hash를 고쳐 승인된 것처럼 만들면 안 된다. 교수님 MMDL/V8은 불변이다.

재사용 시 forecast가83,786.558초에서83,478.931초로 줄어드는 것은 baseline stage의
원래 wall time 대신 저장 evaluator summary.seconds를 사용하기 때문이다. 가속 실측이 아니다.
이 예상에는 새 설치/복원/다운로드·G7/독립 transfer 시간이 들어 있지 않다.
추가 예산/시간 승인 전 새 GPU/외부 업로드를 시작하지 않으며 계정 Pod 목록0개를 재확인했다.
입력/학습 조건·full10K·네 조건·기본2배 여유를 유지한다.
근거 `analysis_exports/g6_closed_results_20261004/resume_readiness.json`.

## 2026-10-04 17:37 UTC — GPU 실제 사용·4종 smoke 완료·예산 gate 중단·회수/삭제 완료

소유 A40 `qg7krba8ebl10s`는 GPU 오류로 대기한 것이 아니다. 고정 f3 코드의 F0
MMMU60 평가32/60·외부 혼합60 평가11/60과 FL/FM/FV/FML 각각360입력·23업데이트,
별도 프로세스 checkpoint 재로딩이 모두 exit0으로 닫혔다.17:17UTC GPU94%·19,759MiB 등
실제 GPU 사용을 관측했다. 최초 지연은 설치/검사/CPU 입력 준비의 직렬 실행이었다.
새4,096출력/60문항 진단 결과는 과거 V8 32K558/900의 개선 성적으로 비교하지 않는다.

17:21:52UTC controller의 기본2배 학습 여유를 유지한 예산 gate는 필요83,786.558초
(23시간16분)와 남은48,418.661초(13시간27분)를 비교해
`SMOKE_COMPLETE_PILOT_BUDGET_STOP`, `pilot_started=false`로 종료했다.
잔액 소진/GPU 장애가 아니며 full10K 네 조건의 학습·평가는 시작하지 않았다.
안전 여유/데이터량/학습 영역/생성 조건을 자동 축소하거나 새 Pod를 대여하지 않는다.
G6 full10K/G7/독립 전문분야 transfer는 미완료이며 추가 유료 실행은 새 예산/시간 확인이 필요하다.

닫힌 inner46파일1,151,423,673bytes·outer19파일113,109bytes·setup4파일7,183bytes·
launch3파일148,115bytes를 로컬 회수해 원격 SHA256/크기·동결 코드/11개 입력 연결·
모든360 ID/토큰/업데이트 노출·4종 checkpoint 실제 tensor·fresh reload 및 예산 산술을 검증했다.
최종 CPU 검증은 native f3 cwd와 실제 import 경로를 확인했다. 초기 inline 검증의 cwd 우선
import 문제를 바로잡아 네 조건 모두 다시 검사했다. 기존 입력/원문/metadata는 불변이다.
기존에 검증한10K/두 cohort/smoke 입력·이미지와 원본 코드도 함께 로컬에 보존한다.
base 가중치·venv/HF cache·원본 benchmark Parquet를 회수했다고 주장하지 않는다.
최종 원장 `analysis_exports/g6_closed_results_20261004/local_receipt.json`,
검증 `final_closed_validation.json`, 원격 파일목록 `remote_closed_manifest.json` 참조.

로컬 검증 후17:37:48UTC 해당 Pod 삭제204→조회404→계정 Pod 목록0을 확인했다.
작업용30GB container/160GB workspace는 폐기했으며 별도 Network Volume은 생성하지 않았다.
보존 결과는 로컬에 남고 해당 Pod의 추가 과금은 중단됐다. 전체 실험 완료는 아니다.
교수님 MMDL·V8 558/900·실행 f3 및 MMDL_test 게시본은 변경하지 않았다.

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
