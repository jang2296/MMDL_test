# Claude 인계 — 로컬 RL+SFT 검증 중단 상태 (2026-11-05 10:44Z)

사용자가 주간 사용량 부족으로 상태 저장 후 중단을 명시했다. 목표는 미완료다. 추가 실행은 사용자 지시에 따른다. 병렬 에이전트6개는 중단했고 마지막 GPU 조회에 실행 프로세스는 없었다. RunPod·유료 API·실제 Qwen 가중치 갱신/저장·commit/push는 하지 않았다.

## 먼저 읽을 파일

- 작업공간 `/mnt/c/projects/multimodel`; `AGENTS.md`, `docs/PROJECT_STATUS.md` 최상단.
- `claudedocs/review_rl_sft_hybrid_v4_20261105.md` §12.5–12.9.
- private root `/home/jang2/mmdl-artifacts/self-note-single-20261105/`의 `usage-constraint-checkpoint.json`, `progress.json`.
- feature branch `feat/mmmu-baseline`. 기존 수정이 많으므로 reset/전체 stage 금지. 교수님 `MMDL/`은 건드리지 않는다.

## 완료와 미완료

1. F4 통합 CPU149검사 및 300→3200 연결 CPU57검사는 통과했다. 실제 GPU 전체 성공과 구별해야 한다.
2. 후보11751개 자동 중복 검색은 종료했다. 공간 후보437family 중362개 직접 대조가 닫혔고75개(part15/16/18)는 미완료다. 중단된 부분 결과는 확인 후 재사용하되 완결 증거 없이 PASS로 쓰지 않는다.
3. 최신 적용 원장은 `data-audit/replacement-no-pdf-v3/adjudication-reuse-v5/capacity.json`이다. 닫힌 part10/17 등의 후속 결합, 최종3500 목록 고정 및 품질 gate는 아직 필요하다.
4. 잠정v7은 OLD1320+NEW2180, 같은 의료 C4 안에서 신규SLAKE96/NCT204다. 미확정 SLAKE는 제외했다. 11출처 품질 검수 packet은 `data-supply/final-selection-preparation-v1/source-prefix-packets-v7/`에 있으며 최종3500 승인 자료가 아니다. 최종 목록과 전체 source queue·검수 prefix·문항·이미지·규칙이 같아야 재사용할 수 있다. ScienceQA에 그림 없이 답할 수 있는 문항이 발견돼 품질 문제도 해결 전이다. Root ChartQA는 first5 관찰만 저장했고 최종 판정하지 않았다.
5. 실제 NF4 첫2: ScienceQA 정답119토큰, NCT 오답245토큰/노트269토큰. ScienceQA 풀이 검토PASS, NCT 노트UNKNOWN/LEAK라 재풀이·A2·B에 넣지 않았다. 전체 오답→정확한 노트→재풀이 정답→RL/A1/A2/B 기울기 성공은 아직 없다.

## 우선 해결할 실제 RL 오류

`single-next10-f4-readiness/f4/close-first2.log`, `runtime/dry-run-2.json`, `close-first2-exit.json`을 확인한다. 첫 확률 재생에서 NonFiniteStop으로 exit2; component 완료0, optimizer/scheduler step0, 저장0, 전체 가중치 불변이다. 오류 문구에 KL이 있지만 실제 호출은 reference=false여서 full KL 전이다.

같은 원래 토큰으로만 수행한 `replay-diagnostic/actual-01/diagnostic.json`은 exit0, policy 불변이다. NCT 전체 문장 재생은 action12에서 support와 log-q가 달랐고 차이는0.1209308654였다. head1/autocast로 해결되지 않았다. 수동 cached 재생은 action4에서 원래 토큰472가 support에서 빠졌다. 따라서 NF4 GEMV/GEMM 차이만을 확정 원인이라고 하지 말고, 실제 collector의 prepared tensors·cache·forward 경로와 fresh NF4 기저 동일성을 먼저 분리 진단한다.

추가로 확인된 수정 필요점:
- `rl.policy_sha256`가 설치된 L/M/V만 묶어 원래 frozen NF4 packed weights/quantization-state 동일성을 증명하지 못한다. 기존 bundle hash helper를 재사용해 새 protocol에 결합한다. 원래 F4 기저 fingerprint는 저장되지 않아 소급 인증할 수 없다.
- `train_v4.backward_single_block`의 preflight는 no-grad만 검증하고 실제 미분하는 q/support를 재검증하지 않는다. grad-enabled forward 전체를 첫 backward 전에 검증하고 실제 backward 경로에도 검증이 연결돼야 한다.
- 이 두 수정과 수치 정렬은 아직 구현·검증하지 않았다. 중단 직전 확인한 single.py/rl.py/train_v4.py/collect.py는 F4와 byte동일했다.

확률 tolerance 확대, 원래 action 억지 수용, mask 교체, 임의 raw-softmax 목적함수 전환, cached KV detach로 통과시키지 않는다. 새 수치 수정 후 기존 CPU 반례와 실제 grad-enabled q/support·손실·기울기·메모리를 확인한다. 동결F4를 고치지 않고 새 소스/동결본으로 분리한다.

## 재개 순서와 제한

사용자 지시를 받은 뒤 실제 RL 오류부터 해결하고, 동일 오답의 노트/재풀이/A2·B 기울기와 PG/A1을 각각 확인한다. 자료 직접 대조/품질/최종3500 영수증을 닫고 실제 메모리별 병렬 처리와 선정3500 campaign 준비를 검증한다. 짧은 NF4 실행을 8192토큰 최대 병렬 수나 RunPod BF16 성공으로 보고하지 않는다.

각 문항 첫 풀이·노트·재풀이 최대1회, 여러 답 중 선택 금지. 완료 답 재생성 및 실패 원장 덮어쓰기 금지. benchmark는 제외 신호로만 사용하고 불명확한 중복 family는 제외한다. 보호된7파일과 MMDL 제출본은 수정하지 않는다. 로컬은 무갱신 검증이며 RunPod는 사용자 별도 명령 전 보류한다.
