# MMDL 프로젝트 지침 — Qwen3-VL-4B 평가·실패 분석·오답노트 RL 개선·재현

> **이 파일은 `AGENTS.md`(Codex용)와 `CLAUDE.md`(Claude Code용)에 byte 동일하게 둔다.**
> 한쪽만 고치지 않는다. 수정 후 `cmp AGENTS.md CLAUDE.md`로 동일성을 확인한다.
>
> 2026-10-30 축약본이다. 이전 6,498줄 원본(2026-10-28 마지막 갱신, SHA256
> `2d65b4f4c2f6a296b71ee6f619da3358557cc24fdaee487ea5e1f94ecd89d403`)은 Git 제외 경로
> `analysis_exports/agents_md_backup_20261030/AGENTS_full_20261028.md`에 그대로 보존했다.
> 날짜별 세부 수치·SHA·PID·실패 원장은 그 원본, `docs/PROJECT_STATUS.md`, 각 private 원장이 정본이다.
> 이 파일은 **규칙 + 현재 상태 + 파이프라인 설명 + 요약 이력**만 담는다.
>
> 표기: `[수업]` 첨부 수업 규정 · `[사용자]` 사용자 지시 · `[설계]` 팀 구현 정책 ·
> `[외부]` 공식 기술 근거 · `[실측]` 저장 결과로 확인한 사실 · `[미검증]` 가설/계획.
> 팀 설계를 수업 규정이라고 말하지 않는다. 문서·코드의 존재는 실행 성공 증거가 아니다.

## 목차

- §0 읽는 순서와 우선순위
- §1 프로젝트가 하는 일
- §2 절대 규칙 (가장 먼저 확인)
- §3 오답노트형 자기개선 RL 파이프라인
- §4 현재 상태 (2026-10-31 기준)
- §5 저장소·게시 정책 (MMDL / MMDL_test / 상위 작업공간)
- §6 MMMU baseline 결과와 고정 평가 조건
- §7 평가 프로토콜 상세 (P0·생성·seed·파서·기록)
- §8 하드웨어·환경·재현
- §9 manifest·재개·비용·회수·정리 gate
- §10 코드·모델·데이터·결과 저장 위치
- §11 1차 평가 실측 원장 요약 (R1/R2)
- §12 이력 요약 (2026-09-21 → 2026-10-30)
- §13 반복 실패에서 얻은 교훈
- §14 진행·제출·발표
- §15 근거 자료와 외부 출처
- §16 이 문서의 유지 규칙

---

## 0. 읽는 순서와 우선순위

1. **§2 절대 규칙**을 먼저 확인한다. 사용자가 명시적으로 해제하기 전까지 어떤 작업도 이를 넘지 않는다.
2. **§4 현재 상태**에서 지금 살아 있는 실행·다음 단계를 확인한다.
3. 작업 대상이 오답노트 RL이면 **§3**과 `fine_tuning/notebook_rl/README.md`,
   설계 정본 `claudedocs/notebook_rl_redesign_20261026/`(research·plan·architecture.html)을 읽는다.
4. 평가/제출 작업이면 §5–§9를 읽는다.
5. 최신 사용자 지시가 이 문서와 충돌하면 사용자 지시가 우선한다. 단 §2의 금지 사항은
   사용자가 그 항목을 직접 해제하기 전까지 유지한다. 승인은 범위가 정해진 것이며 다음 단계로 확대하지 않는다.
6. 오래된 "진행 중" 표기를 현재 상태로 단정하지 않는다. 실제 코드·로그·manifest·프로세스를 확인한다.

작업 시작 시: 적용 지침, Git 루트/상태(`git status`, `git branch`), 관련 README·설정·테스트,
`docs/PROJECT_STATUS.md`를 확인한다. 기존 작업을 reset하거나 덮어쓰지 않는다.
상태·결정·검증 명령·증거·다음 행동은 `docs/PROJECT_STATUS.md`에 기록한다.
변경 없는 전체 재독, 대형 로그/900응답의 대화 출력, 자동 에이전트 토론/무한 리팩터링은 금지한다.
같은 차단 원인의 수정·검증은 최대 3회다. 장시간 작업은 실행 장비의 프로세스/로그와 드문 확인으로 운영한다.

---

## 1. 프로젝트가 하는 일

`[수업]` MMDL(멀티모달 딥러닝) 팀 프로젝트. 팀명 **Team 2** (장우석, 정예찬, 김기원, 김정현).
지정 모델 **Qwen3-VL-4B-Instruct**에 대해 다음 순서를 수행한다. [S3 pp.4–5; S4 pp.14–20]

| 단계 | 내용 | 현재 |
|---|---|---|
| A. Baseline 평가 | MMMU validation 30과목×30=900문항, 공식 비교값 67.4 | **완료**: V8 채점 558/900 (62.00%) |
| B. 실패 분석 | 생성 잘림·파싱·시각 관측·지식/규칙 적용 병목 분리 | 진단 완료(내부 보고서) |
| C. 외부 데이터·학습법 | 벤치마크와 분리된 프로그램 생성/공개 데이터 | 지도학습 시도는 개선 실패 → RL로 전환 |
| D. 실제 fine-tuning | 실제 파라미터 업데이트(LoRA/QLoRA/adapter 등) | **오답노트형 자기수정 RL 구현·검증 중** (§3) |
| E. 동일 조건 재평가 | baseline과 같은 평가·채점으로 비교 | 미완료 |
| F. 최종 평가 | MMMU test, MMMU-Pro Standard 10-way | 미착수 (취득/hash만) |

- Fine-tuning은 **실제 파라미터 업데이트**여야 한다. 프롬프트/RAG/앙상블/파싱 개선만으로 완료하지 않는다.
- 원본 구조 유지가 기본이며 구조 변경은 필수가 아니다. 초파리 신경망·압축·트레이딩 등 다른 주제를 섞지 않는다.
- 현재 사용자의 최우선 목표: **정확한 오답노트 → 검증된 내용의 실제 RL → 경로 선택 → 저장·새 프로세스 재로딩
  → 새 문항 비교**로 이어지는 오답노트형 자기개선 학습 파이프라인 완성. 부분 판정 학습·정답 재시도·
  가중치 변경만으로 완료라고 하지 않는다.

---

## 2. 절대 규칙

### 2.1 학습 방식 — 지도학습 금지

`[사용자 2026-10-16, 이후 반복 재확인]`

- **지도학습(SFT)을 하지 않는다.** 특히 상위 모델(GPT/Gemini/Claude/Gemma 등)의 풀이·설명을 정답 문장으로
  모방시키는 SFT/distillation, teacher rationale, cross-entropy target, DPO 대체를 절대 쓰지 않는다.
- 모델 자신이 생성한 문장을 정답으로 고정해 외우게 하는 STaR/RFT식 SFT도 설계에서 제외한다.
- 학습은 **독립 검증기 보상 기반 강화학습(RL)**만 사용한다. 정답은 검증기(private verifier)에만 전달하고
  모델 입력(actor payload)에는 gold 문자열·기준 풀이·오류 목록·검증 결과의 정답값을 넣지 않는다.
- 정확한 명칭은 "추가 SFT와 교사 모델 풀이 모방을 사용하지 않는 독립 검증 보상 기반 자기수정 RL"이다.
  비공개 정답·프로그램 검사로 보상을 주므로 순수 "자기지도학습"이라고 부르지 않는다.
- 학습 라벨은 프로그램 생성 또는 사람이 만든 것만 사용한다. AI 검토는 `human_verified=false`로 기록하며
  사람 인증으로 표시하지 않는다.
- 이유: 과거 지도학습 시도(G6 FL/FM/FV/FML, TabMWP A/B)가 MMMU 또는 외부 지표를 개선하지 못하고
  오히려 낮췄다(§12). 수업도 벤치마크 학습을 금지한다.

### 2.2 데이터 — 벤치마크 오염 금지

- **MMMU / MMMU-Pro의 모든 split과 모든 파생물**(paraphrase, 수치·이미지만 바꾼 변형, 전사, 응답, 해설)은
  학습·증강에 사용하지 않는다. `BENCHMARK_DERIVED`, `training_allowed=false`, `public_upload_allowed=false`를 전파한다.
- "validation으로 tune"은 정답 학습 허용이 아니다. [S4 pp.14,16,21]
- 오염 검사에는 벤치마크 정보를 **제외 신호로만** 쓰고 학습기로 전달하지 않는다.
  검출기·임계값은 완화하지 않는다. 경고 후보는 실제 문제·이미지 대조로 DISTINCT/DUPLICATE/UNRESOLVED를
  별도 원장에 기록하고, DUPLICATE·UNRESOLVED는 같은 family까지 blacklist로 제외한다(2026-10-17 사용자 승인).
- 학습/평가 문항의 출처·구조(전체 multigraph 등)·이미지 hash 분리를 확인한다.
- 학습 결과에 따라 표본을 몰래 교체하지 않는다. 사전 고정한 순서·ID·seed를 유지한다.

### 2.3 수업 고정 조건 — 변경 금지

`[수업: S1 §§0–4; S2; S4 pp.14,16,21]`

```yaml
model_id: Qwen/Qwen3-VL-4B-Instruct
model_revision: ebb281ec70b05090aa6165b016eac8ec08e71b17
dtype: bfloat16            # baseline 평가는 BF16·비양자화·원본 가중치
mmmu_dataset_id: MMMU/MMMU
mmmu_dataset_revision: 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68
assignment_split: validation
assignment_subjects: 30
assignment_samples_per_subject: 30
assignment_samples_total: 900
assignment_report: reports/mmmu_baseline.md
assignment_deadline: "2026-09-28 23:59 (수업 공지 기준)"
```

- 공식 Qwen 배포본만 사용한다(커뮤니티 재업로드·GGUF·AWQ는 baseline 대체물이 아니다).
  processor/tokenizer/chat template도 같은 revision.
- 30개 subject config를 각각 로드한다. 로드 오류 과목을 버리거나 일부만 제출하지 않는다.
- 과목별 accuracy 30개와 macro average(과목당 30개이므로 = 정답 수/900)를 기록한다.
- 공식 비교값 **67.4**는 과제 지침의 값이며 우리 측정값이 아니다. 차이 진단은 **1000자 이내**.
- 한 명령으로 재실행하고 모델·데이터 위치를 인자/환경변수로 바꿀 수 있어야 한다.
- 4090 24GB에서 실행 가능한 경로를 제공한다(24GB가 강제 최대치라는 뜻은 아니다). [S1 §1.3]
- MMMU-Pro는 수업 지정 **Standard 10-way**만 기준으로 한다(vision-only로 대체 보고 금지). [S5 p.11]
- QLoRA/NF4 학습은 수업이 허용한 학습 방식 예시이지만 baseline 평가의 BF16 무양자화 규칙과 구분한다. [S4 p.18]

### 2.4 저장소·코드 위치

- **교수님 제출용 `MMDL/` 폴더와 `jang2296/MMDL` 저장소는 절대 수정하지 않는다**(사용자가 명시 요청한 제출
  작업 제외). 실험 코드·결과·이미지를 MMDL로 동기화하지 않는다.
- 실험 코드는 `fine_tuning/` 안에만 둔다. 오답노트 RL은 `fine_tuning/notebook_rl/`.
- `fine_tuning/train.py`, `fine_tuning/modules.py`, `fine_tuning/protocol.json`은 과거 manifest가 SHA로
  고정하므로 수정하지 않는다.
- 실행 중인 동결(frozen) 소스 복사본은 hotpatch하지 않는다. 수정은 상위 소스에서 하고 새 동결본/새 protocol로 분리한다.
- 비밀키·가중치·체크포인트·HF/benchmark 원문·이미지·대형 결과·강의 원본은 어떤 공개 저장소에도 올리지 않는다.
- 중복 구현·빈 학습/증강기·가짜 PASS·임시 v2/final/new 파일·불필요한 프레임워크를 만들지 않는다.
  기존 helper를 먼저 재사용한다. 전역 환경·OS/WSL/driver·다른 프로젝트를 임의 변경하지 않는다.

### 2.5 결과 보고의 정직성

다음을 서로 섞지 않는다. 섞어서 보고하는 것이 가장 자주 지적된 실수다.

| 구분해야 할 것 | 설명 |
|---|---|
| fixture/CPU 검사 통과 ↔ 실제 Qwen GPU 실행 | 작은 무작위 모델·작성 fixture는 자기개선 증거가 아니다 |
| 노트 생성 ↔ 정확한 노트 ↔ 노트 내용의 학습 | 생성·형식 통과·strict 정확성·실제 기울기 기여는 각각 다르다 |
| 답 보상 update ↔ 노트(오류·수정) 학습 | 가중치 변경이 답 보상/KL에서 왔으면 노트 학습이 아니다 |
| 재시도 정답 회복 ↔ 가중치 자기개선 | 같은 부모의 재시도 회복은 독립 첫 시도 개선이 아니다 |
| 부분 판정 학습 ↔ 전체 오답노트 성공 | bounded claim 정확성을 전수 오류 설명으로 확대하지 않는다 |
| UNKNOWN/관측 누락 ↔ Vision 정상 | 미확정은 정상 Vision 인증이 아니다 |
| 독립 관측 오류 ↔ 내부 Vision 모듈 단독 책임 | 운영 분류이지 인과 인증이 아니다 |
| AI 검토 ↔ 사람 검증 | `human_verified=false` 유지 |
| 산술 예상 ↔ 실측 재채점 | 예: "24개만 바로잡으면 53.67%"는 가정 계산 |
| 실행 종료 ↔ 유효한 전수 판정 | 시간한도·BUDGET_STOP·실패는 그대로 보고 |
| ON 단독 update ↔ OFF-only/양 경로 검증 | 혼합 update는 Vision 동결 경로 증거가 아니다 |
| 중간저장 ↔ 전체 완료/새 process reload | 중간 checkpoint는 재개 지원·최종 완료가 아니다 |

- 검증 기준·보상·채점·한도를 완화해서 통과시키지 않는다. 실패는 원문과 함께 보존하고 덮어쓰지 않는다.
- 과거 결과를 새 protocol로 소급 재표기하지 않는다. 새 scoring/protocol version으로 분리한다.
- 테스트를 skip/disable해 통과시키지 않는다. 실행하지 않은 검사를 통과했다고 하지 않는다.
- "blazingly fast", "100%" 등 근거 없는 표현, 근거 없는 시간·퍼센트를 쓰지 않는다.

### 2.6 실행·비용·클라우드

- 대규모 학습, 새 900문항 평가, cloud 실행, 다른 GPU·고가 GPU 확대, fallback(양자화·해상도·토큰·
  정밀도 자동 축소), 자동 연장·재시작을 **자동으로 시작하지 않는다**. 각각 사용자 승인이 필요하다.
- RunPod 실행 전: 실제 잔액·가격·최초 과금 시각·회수 예약금·GPU/금융 종료 시각·최대 시간을 고정하고
  watchdog을 둔다. 자동충전·결제 설정은 바꾸지 않는다. 약속된 충전은 실제 잔액이 아니다.
- RunPod는 GitHub 고정 commit의 clean clone으로만 실행한다. 로컬 코드/venv/모델/cache 업로드, Pod 수동 패치 금지.
- 실행 중 10분 간격으로 Pod/프로세스/오류/예산을 점검한다. 같은 원인의 무근거 반복 실행 금지.
- 정리 순서: **REMOTE_VERIFIED → LOCAL_VERIFIED → POD_TERMINATED → OWNED_STORAGE_DELETED → CLEANUP_VERIFIED**.
  로컬 수신·hash·재채점 검증 receipt 전에는 삭제하지 않는다. Stop만으로 정리 완료라고 하지 않는다.
- **이 대화/작업이 만든 소유 자원만** 삭제한다. 공유·기존 volume, 다른 프로젝트 자원은 건드리지 않는다.
- 로컬 프로세스 제어는 PID와 startticks·명령·cwd를 대조한 **소유 프로세스에만** 신호를 보낸다.
- 공유 BF16 원본 가중치는 기존 checkpoint 6개(G6 FL/FM/FV/FML, TabMWP A/B)가 의존하므로 삭제하지 않는다.
  격리가 필요하면 non-overwrite 이동 + journal로 복구 가능하게 한다.
- 외부 유료 API·LLM judge 호출은 별도 승인 없이 하지 않는다.

### 2.7 Git

- feature branch에서만 작업한다(현재 상위 `feat/mmmu-baseline`). main/master 직접 작업 금지.
- commit/push는 사용자가 요청할 때만. force push·이력 재작성·공개 범위/계정 설정 변경 금지.
- 상위 저장소에서 `git add -A`/전체 push를 하지 않는다. 명시 파일만 stage한다.
- 공개 게시는 §5의 동기화 스크립트와 검사를 거친다. commit 전 `git diff` 검토.

### 2.8 단일 답 원칙 — 여러 답을 만들고 고르기 금지

`[사용자 2026-10-30, 중요 제한]`

- 평가·보고 점수는 **문항마다 한 번 생성한 답 하나**로만 낸다. 여러 답을 만들어 그중 하나를 고르는 방식은 모두 금지:
  best-of-N, 다수결(self-consistency), 재순위(reranker·verifier·LLM judge로 선택), 정답·검증기 피드백을 이용한 재시도 후 선택, 앙상블.
- 여러 seed·실행·체크포인트 중 좋은 것을 고르지 않는다. 여러 seed 평가는 seed별 정확도의 평균으로만 보고하고 문항별로 seed를 고르지 않는다.
  과제용 점수는 사전 등록한 실행 하나(과제 protocol, seed 3407)다. arm당 평가 체크포인트는 하나다.
- `[설계 해석]` 학습 중 RL 그룹 표본(문항당 n개)은 모든 표본의 보상이 기울기에 쓰이는 학습 신호이며 답을 고르는 것이 아니다.
  오답노트 파이프라인의 검증기 피드백 재시도·회복률은 학습·진단 지표로만 쓰고 모델 정확도로 보고하지 않는다.
  사용자가 다르게 지시하면 이 해석을 고친다.

---

## 3. 오답노트형 자기개선 RL 파이프라인

### 3.1 무엇을 만드는가

모델이 실제로 틀린 풀이를 **스스로 점검·수정**하고, 모든 시도를 종합한 **오답노트**를 작성하게 한 뒤,
독립 검증기가 확인한 수정 행동과 노트 내용에 **RL 보상**을 주어 가중치를 실제로 바꾸는 파이프라인이다.
설계 정본: `claudedocs/notebook_rl_redesign_20261026/research_notebook_rl_redesign_20261026.md`(D00–),
`plan.md`(P00–P23, 요구사항 R01–R18), `architecture.html`. 초기 설계(2026-10-16)는
`claudedocs/research_error_notebook_self_correction_rl_plan_20261016.md`와 `error_notebook_rl_pipeline_20261016.html`.

사용자 의도의 구현(설계 D00):

1. 같은 Qwen 모델이 이미지·문제를 보고 풀이와 답을 생성한다(T1).
2. 별도 검증기가 답 정오와, 검사 가능한 관측(이미지에서 읽은 사실)·연산(규칙 적용)의 모순을 판정한다.
3. 실패한 실제 응답을 문맥에 그대로 남긴 채, 모델이 한 completion 안에서
   `ErrorCheck:`(오류 점검 JSON) → `RevisedSolution:`(수정 풀이)를 순서대로 생성한다(T2/T3).
4. 답이 맞아도 검증된 과정 오류(VIOLATED 관측·연산)가 남으면 재시도 대상이 될 수 있다.
5. 정답 후보가 생기면 모델이 **모든 실제 시도(실패·잘림·성공·형제·조상·피드백)와 이미지**를 보고
   최종 오답노트(final note)를 작성한다(n4).
6. 검증기가 노트의 주장·인용·수정 관계(CORRECTED/PERSISTED/UNKNOWN 등)를 확인하고,
   **실제 생성된 원래 토큰에만** RL 보상을 연결한다.
7. 검증된 관측 오류 근거로 Vision 학습 포함 여부(경로)를 고르고, 실제 update → 저장 → 별도 프로세스 재로딩을 검증한다.
8. 새로운 문항의 첫 시도 성능을 별도로 측정한다. 기능 통과를 정확도 향상으로 바꾸어 말하지 않는다.

용어 바로잡기: "맞을 때까지" = 최대 3단계 안의 재시도(못 맞히면 UNRESOLVED로 종료).
"답지를 보고" = 검증기만 정답과 비교. "찍어서 맞혔나" = 정답인데 검증된 과정 오류가 남았는지 관측.
"풀파인튜닝" = 로컬에서는 하지 않는다(아래 3.2).

### 3.2 모델과 학습 범위

- 모델 `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`.
- 로컬 RTX 5060 8GB는 BF16 가중치(8.88GB)만으로 VRAM(8.55GB)을 넘고, 기존 native HF+vLLM 경로 하한은
  26.5GiB다(2026-10-18 실측). 그래서 2026-10-21부터
  **NF4 양자화 기저 동결 + Vision LoRA + LLM LoRA + 비양자화 Merger 4개**를 학습한다(QLoRA 계열).
  Merger: `model.visual.merger`, `model.visual.deepstack_merger_list.{0,1,2}`.
  설치 trainable tensor 624개(L 어댑터/M Merger/V Vision 어댑터). **full-parameter FT가 아니다.**
- 경로(route)는 독립 관측 검증으로 정한다. 모델 노트의 자기주장으로 정하지 않는다.
  - `VISION_ON`: 독립 검증된 관측 오류 근거 → Vision adapter + Merger + LLM adapter 학습.
  - `VISION_OFF`: 필수 관측이 모두 검증됨 → Vision 동결, Merger + LLM adapter 학습.
  - `UNRESOLVED`/미확정: 학습 진입 금지. 미확정을 Vision 정상으로 처리하지 않는다.
- 별도 trainer·framework·scheduler·다운로드 계층을 새로 만들지 않는다. 기존 `fine_tuning` 경로를 재사용한다.

### 3.3 현재 고정 설정 (로컬 NF4 경로)

| 항목 | 값 |
|---|---|
| 단계·표본 수 | T1 n4 → T2 n2 → T3 n2, 최대 3단계 |
| 출력 상한 | 풀이 1024 토큰(ErrorCheck 포함), 최종 노트 8192 토큰 |
| 문맥 | 24576 (초과 시 원문을 자르지 않고 `CONTEXT_BLOCKED`로 기록) |
| 생성 | 기존 sampling, master seed 3407, 샘플별 seed 식 고정 |
| 보상 | 답 binary(CORRECT 1/기타 0), online repair PG .0625, final note PG .0625, KL .01 |
| 정규화 | 원래 그룹 크기·고정 분모·MIS·원래 n 유지, ErrorCheck/노트 토큰에는 답 PG 없음 |
| update | 최소시험 최대 1 update(2문항), 고정10 최대 5 update(pair 단위) |
| 시간 | 내부 14400초, 외부 14430–14700초 + kill 30초 (실행별 launch 원장 확인) |
| 부모 선택 | `SHA256(f"{key}:{sample_id}")` 오름차순 + 답 첫 등장 우선, native 후보 우선 |

- **재관찰 v3**(`repair-retry-collector-v3`): 첫 eligible 재시도 직전에 같은 정책이 원본 이미지·질문·
  공개 관측 문법만 보고 n1/1024로 한 번 다시 관찰한다. 이전 풀이·정오·gold는 주지 않는다.
  T2/T3와 최종 노트가 이 원문을 **미검증(UNVERIFIED) 문맥**으로 공유한다. 보상·시도·그룹·라우팅·loss에서 제외.
- **문법 실패 응답**: 원문과 logprob가 완전하면 `result=None`/`INVALID_RESPONSE_CONTRACT`로 보존하고
  PG 0으로 원래 그룹에 남긴다. 정상 후보가 없을 때만 실제 원문 SHA를 재확인한 문법 실패 응답 하나를
  다음 부모로 쓴다(`repair-retry-collector-v2`). 원문 누락·무결성 실패는 계속 차단한다.
- **동일 보상**이면 상대 신호 0이다. 가짜 분산을 만들지 않는다. 무신호 문항은 순서대로 소비하고 교체하지 않는다.

### 3.4 상태 머신

```text
DATA_BOUND → INPUT_CHECKED → MODEL_READY
  → T1_GENERATED → ANSWER_AND_PROCESS_AUDITED
  → [RETRY_ELIGIBLE and attempts_remaining]
       (첫 eligible 재시도 직전) FRESH_REREAD n1 — 미검증 문맥
       T2/T3: ErrorCheck → RevisedSolution → FRESH_AUDIT
  → TERMINAL_HISTORY
       정답 후보 있음 → ALL_HISTORY_FINAL_NOTE(n4) → NOTE_AUDIT
       정답 후보 없음 → UNRESOLVED_RECORD (가짜 성공 노트를 만들지 않음)
  → INDEPENDENT_ROUTE
       UNRESOLVED → SAVE_NO_ADMISSION
       ON/OFF → CREDIT_BOUND → PREUPDATE_REPLAY → UPDATE_OR_NO_SIGNAL
  → SYNCHRONIZED → CHECKPOINT_CLOSED → FRESH_PROCESS_RELOAD
  → FUNCTIONAL_RESULT / REPAIR_RESULT
```

종료 규칙: 답이 맞고 검증 가능한 모순이 없으면 그 분기 종료(범위 부족이면 "답 맞음/과정 미확정").
마지막 시도까지 실패하면 실패로 닫는다. CONTEXT_BLOCKED·BUDGET_STOP·OOM·비유한 손실·원문 변조는
원인을 저장하고 종료하며 한도·모델·검증 기준을 자동 완화하지 않는다.
형제 후보의 정답을 다른 부모의 회복으로 세지 않는다.

### 3.5 검증 등급 — F / S / E

| 등급 | 필수 관측 | 이것만으로 말할 수 없는 것 |
|---|---|---|
| **F 기능** | 실제 Qwen 생성 → 실제 보상·원래 토큰 → 유한 기울기 → 파라미터 변경 → 실물 저장 → 새 프로세스 재로딩(optimizer/step/가중치/단일·복수 이미지 forward·생성 일치, 원본 접근 0) | 노트 정확성, 성능 상승 |
| **S 수정 의미** | 실제 native `CORRECTED` 관계·원문 증거·독립 검증, 양성 VERIFIED CORRECTED-only VJP와 전체 signed CORRECTED VJP가 **각각** 유한·비영, 실제 저장 update에 연결 | 내부 Vision 단독 책임, 모든 자연어 추론의 정확성 |
| **E 효과** | 사전 고정한 별도 문항에서 첫 시도·회복·노트 정확성·비용을 동결 평가기로 비교 | 작은 표본을 MMMU 전체 개선으로 일반화 |

- 작은 기능시험에서 성능 상승은 필수가 아니다(사용자 2026-10-26 정정). 그러나 정확한 내용 검증과 실패 보존은 필수.
- 코드 상태 단계: DESIGNED → IMPLEMENTED → TESTED → FUNCTIONAL_VERIFIED → REPAIR_VERIFIED → EFFECT_MEASURED
  (또는 BLOCKED). 각 전환에는 해당 증거가 필요하다.
- E가 개선을 보이지 않으면 "기능 구현 완료, 대규모 학습 효과 미입증"으로 남긴다. 대규모 학습은 별도 승인.

### 3.6 실행 순서

1. 상위 소스 수정 → CPU 회귀(관련 집중 검사 + 연결 검사) · Ruff · 범위 mypy · 독립 검토.
2. 새 동결 소스 manifest(`package-root-vN`, 파일 byte/size/mode) 고정.
3. source-bound CPU 입력 준비/검증(고정10·기존 170 입력 byte 보존·CPU guard·CUDA 미초기화).
4. 기존 NF4 cache 무변조 재사용 검증(save → 별도 PID check, 두 생성 token 동일·68 logprob 차이 0·원본 접근 0).
   이는 베이스 재로딩 검증이지 학습 checkpoint 재로딩이 아니다.
5. **최소시험**: 고정 첫 single + 첫 paired 이미지 2문항, freshF0, 최대 1 update.
6. 종료 CPU 감사(원문 재생) → 실물 624 tensor 검사(`check_payload.py`) → 공개 `local_train reload`의 fresh reload.
7. **고정10**: `--admission verified-functional-repair-exploration-v1`로 freshF0에서 시작(최소시험 가중치를 이어 쓰지 않음).
   기본 strict admission은 유지된다. 이 경로는 실행 허용이지 S/E 인증이 아니다.
8. 같은 입력·초기값·seed의 **답 보상 대조군(answer-only control)** → 각 arm fresh reload.
9. 별도 독립 평가(native 회로 10문항/3위상 등, n1, 평가 중 학습 0) → F0 / FT / control 비교.
10. S/E 판정 후에만 대규모 학습을 사용자에게 제안한다.

### 3.7 코드 위치와 주요 모듈

- `fine_tuning/notebook_rl/README.md`: 현재 상태·인터페이스 표·재현 명령(정본).
- `repair_contract.py`(재시도/최종노트 문법·투영), `repair_credit.py`(online/final 관계 감사·그룹 credit),
  `repair_tokens.py`(원래 토큰 소유권), `losses.py`(answer/online/final 분리 loss, KL 1회),
  `repair_training.py`(fresh 학습 준비), `updates.py`(optimizer step·VJP 검사),
  `repair_replay.py`·`repair_evidence.py`(원문 재생·저장 update 증거), `checkpoint.py`(저장·
  생성 중 AdamW moment CPU 보관 `park_adam_moments`), `local_train.py`·`local_run.py`(minimum/fixed10/control/reload),
  `local_eval.py`·`local_ladder_*`(독립 n1 평가), `rollout.py`(부모 선택·피드백), `notebook_synthesis.py`.
- 실행 CLI 예: `python -m fine_tuning.notebook_rl.local_run minimal|reload ...`,
  `python -m fine_tuning.notebook_rl.local_train ... --repair-notebook --admission ...` (정확한 인자는 README).
- 폴더 단독 실행(`mmdl`/`experiments` 없이)이 가능해야 한다.

### 3.8 데이터

- 현재 학습 자료는 프로그램 생성 합성 그래프/트리/DFA/최단경로/회로 문항(고정 10문항, 15 이미지)이며
  정답은 private verifier만 가진다. benchmark 파생 없음. `human_verified=false`.
- 별도 평가 자료: native 회로 10문항/3위상(기존 AI 검토 재사용, 새 시각 검토 0), 학습 문항과 출처·이미지·구조 분리.
- 데이터 권한·제외·분리 검사는 재사용하되 새 오염 전수검사를 했다고 과장하지 않는다.
- 증강은 의미 보존을 검증한다. crop/회전/반전이 글자·축·방향·화학 구조·보기 참조를 바꿀 수 있으므로
  무조건 적용하지 않는다. 차트/도형은 수치·구조를 먼저 만들고 렌더링·정답을 프로그램으로 검증한다.
- 추론용 `device_map='auto'`를 학습용 offload로 재사용하지 않는다. effective batch가 같아도 token loss 정규화·
  샘플 순서·update 수·scheduler가 다르면 같은 학습이 아니다. 데이터량/step이 함께 바뀌면 교란 요인으로 보고한다.

### 3.10 compact 오답노트 계약 (`repair-note-compact-v1`, 2026-10-30)

- 입력: 문제·보기 + 모든 시도의 원문(번호 1..N, 재시도는 "retry of attempt k") + 이진 정오(CORRECT/NOT CORRECT/
  NOT GRADED) + 미검증 재관찰 + 원본 이미지. 상태 코드·정답·오류 목록·구체 예시 값은 넣지 않는다.
- 출력: `{"errors":[최대4 채점], "vision_misread":"YES|NO|UNSURE", "lesson":"…"}`. 항목은
  `OBSERVATION{attempt,item,correct}` / `NO_FINAL_ANSWER{attempt}` / `WRONG_STEP{attempt}`.
- 채점: 그 시도의 사실이 fresh 감사에서 VIOLATED이고 제안 값이 verifier 1-사실 검사에서 맞으면 +1, 원래 맞았던
  사실을 틀렸다고 하면 −1, 위치만 맞으면 0; NO_FINAL/WRONG_STEP은 실제 결과·TRACE 검사로 ±1; vision 자기판단은
  독립 strict_routing과 일치 +1/반대 −1. 노트 보상 = 합/5. lesson은 무보상·credit 제외.
- 학습 경로는 여전히 독립 판정(strict_routing)이 정한다. 노트의 vision 판단은 검증·보상 대상일 뿐이다.
- 증거: S_note = 양의 advantage이면서 검증 통과 항목(또는 vision 정답)이 있는 노트 토큰의 VJP 비영
  (`s_definition=verified-note-claim-v1`). v2의 CORRECTED 전용 S와 합치지 않는다.
- 주의: NO_FINAL_ANSWER는 텍스트만으로 확인되므로 S를 시각 학습으로 해석하기 전 항목 유형별 검증 수를 본다.
  현재 데이터가 거의 VISION_ON이라 vision 자기판단이 "항상 YES"로 쏠릴 수 있다.

### 3.9 지금까지 확인된 사실 (요약)

- **F는 여러 번 실제 통과**: V58(판정 단위 보상, signed assessment-only 기울기 비영), V59 v6·frozen8·source10 최소시험
  (update1·L252/M24/V48 변경·624 저장·fresh reload). 단, 대부분 학습 신호는 **답 보상**에서 왔다.
- **S는 아직 미확보**: online/final의 signed·positive native CORRECTED VJP가 모든 실제 update에서 0.
- 최종 노트 반복 실패 패턴: `success_sample_id` 누락, `claims=[]`(허용된 기권이지만 검증 주장 0),
  자유 lesson의 거짓 주장, e4/e8 방향 관측 오류를 정답 후에도 유지, 정상 항목에 대한 거짓 교정 주장.
- 전체 이력·성공 재시도·재관찰 원문은 실제 입력에 무손실 전달됨을 여러 번 확인했다. 전달 누락이 원인이 아니다.
- 많은 이력에서 검증기가 인정할 수 있는 descendant 교정 기회(VIOLATED→COMPLIANT) 자체가 0이다.
  형식만 고쳐도 S가 생기지 않는다.
- 새 문항 효과(E)는 2026-10-21 로컬 비교에서 개선 미관측(첫 시도 F0/NFT/CFT 모두 5/10, 최종 모두 8/10).

---

## 4. 현재 상태 (2026-10-31 기준)

`[실측]` 마지막 기록 시점. 실행 중 표기는 반드시 실제 PID/로그로 재확인한다. 상세는 `docs/PROJECT_STATUS.md` 최상단.

- **사용자 결정(2026-10-30)**: 학습 데이터는 기존 공개 데이터셋(HF 등)만, 자체 생성 금지(저품질 유입 방지). 오염 = MMMU/MMMU-Pro와
  동일한 문제 학습 금지. RunPod는 **최대 3,000문항 증명 1회**, 결과 마감 **2026-11-17**. 학습은 오답노트 파이프라인 1개 arm(Vision 동결),
  평가는 혼합(동시16 새 protocol F0·FT seed 3개 짝 비교 + 과제 protocol 동시2 FT 1회). §2.8 단일 답 원칙.
- **데이터(로컬 고정 완료)**: `hf-pool-v3` — HF 학습 split 10개 출처, MMMU/Pro 전 split 대비 3단계 분류(T1 제외, T2 전부·부족 과목 T3 실제
  이미지 대조 검토, DUPLICATE 0), 출처별 품질 수용 표본 검사, `freeze-v2/dataset_manifest.json`(sha `4ff8f945…`) + `vault-v1`(union receipt).
  학습 적격 3,418(eligible_ids가 정본), dev 402 / sealed 209. 과목 공백(회계·경영·마케팅·심리·예술·디자인·공학 5과목 등)은 채우지 않음.
- **파이프라인 추가(CPU 시험)**: 슬롯 없는 문항용 정답 검증 경로(`--answer-verified-route`, Vision 동결, ANSWER_VERIFIED 기록), 최종답 보상
  (`--finalization-reward`), 평가 정렬 T1(`p0-eval-aligned-v1`, P0 + V8 파서), `--retry-out-of-choice`, 통과율 사전필터·hard-negatives,
  scale(배치·vLLM·BF16 정책·update당 K·k마다 저장·감사 단계 VJP·이어하기), 증명 선택 규칙 `pool-proof-selection-v1`, Pod frozen-approval 모드.
- **실제 GPU(로컬 NF4, 새 데이터)**: v15 최소시험 F 통과(update1, L252/M24/V0, 새 프로세스 재로딩·624텐서 PASS). 고정10 노트 arm update5·재로딩 PASS
  (S_note·S_text false). 대조군 update5·재로딩 PASS. 풀 평가는 결함 2건(체인 인자 경로, guard 아래 lineage 읽기)으로 실패 → 수정·회귀시험 후
  **동결 v16(`proof-freeze-v16/package-root-v16`)으로 로컬 전체 재실행 중**(`local-test-v4/chain-v16.sh`: 최소·고정10·대조군·평가12·GPU 통과율).
- **RunPod 준비(CPU만, RunPod 미실행)**: `fine_tuning/notebook_rl/pod/`(volume·data bundle·job 승인 sha·start/watchdog·chain 단계 stage1/pilot-300/main·
  eval 단계·회수·비용), `env/requirements-proof.lock`, Fable 감사 2회(1차 FAIL→수정, 2차 PASS). 다음: v16 승인 export·번들·MMDL_test 동기화 후
  사용자 승인 게이트(G0 push, G1 volume, GD 데이터 업로드, GE 평가 데이터, G2/G3 1단계).
- **CPU 전체 시험**: 2,738개 중 36개 실패는 10-30 이전 동결본에서도 같은 legacy 결함(증명 경로 미사용, 미수정).
- 교수님 MMDL clean (HEAD `197838c11557911a14a46d87b7f52fda0cac5b3c`), V8 558/900 불변. cloud/API/게시 없음.

---

## 5. 저장소·게시 정책

| 위치 | 용도 | 원격 |
|---|---|---|
| `C:\projects\multimodel` (`/mnt/c/projects/multimodel`) | 내부 원본 작업공간: 지침·docs·claudedocs·실험·원본 자료 | 상위 Git(직접 push 금지) |
| `C:\projects\multimodel\MMDL` | 교수님 제출용 독립 checkout | `jang2296/MMDL` (default `feat/mmmu-baseline`) |
| `C:\projects\multimodel\MMDL_test` | 실험 코드 게시용 독립 checkout | `jang2296/MMDL_test` (feature branch) |
| `analysis_exports/` | 원본 결과·회수본·백업·내부 도구 (Git 제외) | 없음 |
| `/home/jang2/mmdl-artifacts/` | 대형 결과·체크포인트·private 원장 | 없음 |

### 5.1 MMDL (제출본)

- 제출 보고서 정본은 **`reports/mmmu_baseline.md`** 하나(SUBMISSION_TEMPLATE 8절 형식).
  `assignment/assignment1.md`는 링크만. 공개 루트 `Assignment_1.md`는 제외하며 상위의 동명 파일은 과거 기록이다.
- 공개본에는 `AGENTS.md`/`CLAUDE.md`, `docs/`, `claudedocs/`, 강의 PDF, 내부 감사/실험, 비밀키·가중치·
  MMMU 원문/이미지/대형 응답을 넣지 않는다.
- 순서: 상위 수정 → CPU 검사 → `python -m scripts.sync_submission --sync` → `--check` →
  MMDL에서 CPU/공개파일 검사·`git diff --check` → feature branch commit → 정상 push →
  파일 목록·SHA256·실행권한과 원격 HEAD·clean status 대조. MMDL에 독자 수정이 있으면 먼저 상위에 반영.
- 최신 게시 `197838c` (2026-09-28): 소요시간 표기만 변경. 평가 설정·파서·점수 불변.

### 5.2 MMDL_test (실험)

- 실험 코드·설정·문서와 필수 자체 생성 CPU fixture만 게시한다. `python -m scripts.sync_test_repo`로 복사/검사.
- 새 RunPod는 MMDL_test 전체 commit SHA의 clean checkout·고정 lock으로 받는다. 수동 overlay 금지.
- 원문·전사·정답·가중치·실행 결과는 게시하지 않는다.
- 2026-10-21 이후 로컬 NF4 실행은 private 동결 소스 복사본(`package-root-vN`)으로 하며 게시하지 않았다.

---

## 6. MMMU baseline 결과와 고정 평가 조건

### 6.1 최종 baseline (V8 채점)

| 실행 | protocol | 출력 상한 | 이미지 pixel budget | V8 정답 | 정확도 |
|---|---|---|---|---|---|
| R1 | `mmmu-val-v1` (Transformers, batch1) | 2048 | 262144–1310720 | 478/900 | 53.11% |
| R2 | `mmmu-val-fast-vllm-32k-continuous-v1` (vLLM 동시2) | 32768 | 262144–1310720 | 554/900 | 61.56% |
| **B2** | `mmmu-val-official-vllm-b2-v2` | 32768 | **1003520–4014080** (Qwen 공식) | **558/900** | **62.00%** |

- 제출 baseline은 사전 선택한 **B2 + `team-final-answer-v8`** = **558/900 (62.00%)**, 공식 67.4와 −5.40pp.
- 새 평가 진입점 `configs/eval/mmmu_val_v8.yaml`. B2와 protocol ID/채점기만 다르다.
- 원래 채점(공식 MMMU 규칙 기반, 무작위 추측 제거): R1 427 (47.44%), R2 459 (51.00%), B2 557 (V4 기준).
- 채점기 이력: V2(감사 진단용) → V4(명시적 최종답 추출 + 공식 open evaluator) → V5 → V6 → V7(**REJECTED**,
  46개 false negative) → V8 확정. 원본·각 버전 결과는 덮어쓰지 않고 보존.
- V8 집계 `results/scoring-v8-20260924/`, 결과/HTML `analysis_exports/mmmu_20260924/team_final_v8/`,
  설명 `claudedocs/mmmu_v8_pipeline_20260924.md`. 내부 열람본 `results/P1/V8_*.html|pdf`(게시 안 함).
- 이미지 확대 효과(554→558, +61/−57)는 seed·실행 차이와 함께 바뀐 관측이며 인과효과로 단정하지 않는다.

### 6.2 고정 조건 (B2/V8)

- 모델·revision·BF16·비양자화·미학습. 30과목×30, 실제 유형 객관식 847 / 주관식 53.
- 생성: `do_sample=true, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
  presence_penalty=1.5, num_beams=1`, `max_new_tokens=32768`, `max_model_len=40960`.
- vLLM 0.11.0, 연속 처리(continuous refill), 최대 동시 요청 2, eager, prefix cache false.
- 프롬프트 P0(§7), master seed 3407 + 샘플별 seed.
- 이 조건을 바꾸면 새 protocol ID로 분리한다. 같은 이름으로 섞지 않는다.

---

## 7. 평가 프로토콜 상세

### 7.1 P0 프롬프트

근거: Qwen3-VL Technical Report arXiv:2511.21631v2 부록 B.1 p.34, Qwen `evaluation/mmmu/run_mmmu.py`의
`build_mmmu_prompt`, VLMEvalKit `_build_mmmu_prompt`. 상세 commit·위치는 `docs/SOURCE_REVIEW.md`.

객관식: 모든 보기를 원래 순서대로 `A. 내용` 한 줄씩.

```text
Question: {question}
Options:
{options}
Please select the correct answer from the options above.
```

주관식은 `Question: {question}`만. 유효한 원본 hint가 있을 때만 `Hint: {hint}\n`을 앞에 붙인다.
이미지 content를 참조 순서대로 먼저 넣고 P0 텍스트를 마지막에 둔 단일 user message에
지정 processor의 `apply_chat_template(add_generation_prompt=True)`를 적용한다.
추가 system/few-shot/CoT/JSON 강제/도구/대화 이력 없음. 이미지 누락·문자열 대체·이중 resize 금지.

### 7.2 seed

```python
import hashlib
value = hashlib.sha256(f"{master_seed}:{sample_id}".encode()).digest()
seed = int.from_bytes(value[:8], "big") % (2**31)  # master_seed=3407
```

팀 정책이다. Qwen 내부 난수 소비 순서와 동일하다고 주장하지 않는다. 매 샘플 Python/NumPy/Torch RNG 설정.

### 7.3 파서·채점

- 공식 MMMU `mmmu/utils/eval_utils.py` 계열. 무작위 추측 fallback은 제거하여 `NO_PARSE`=오답.
  빈답/파싱 실패도 완료 추론이면 900 분모에 포함. OOM/입력 오류/누락은 시스템 실패(INCOMPLETE).
- `presence_penalty`는 실제 적용을 검사한다(vLLM native generated-only / Transformers custom generated-only).
- Qwen 공개 코드의 LLM 보조 판정은 사용하지 않는다. 외부 LLM judge는 승인 없이 쓰지 않는다.
- 파서 수정은 합성 회귀 → 저장 raw 900 전체 재채점 → old/new 차이·PARSED 회귀 확인 후 새 scoring version으로.

### 7.4 기록

각 샘플: ID·과목·유형·질문·원래 보기·이미지 참조·실제 prompt/messages·seed·입력 hash·**raw response 전체**·
추출답·gold·정오·파싱 상태·종료 사유·token 수·시간. `model_explanation`은 실제 출력 원문 구간/offset만.
Codex/외부 모델이 답/풀이를 채우지 않는다. 공개 `results/<run_id>/`에는 summary·30과목 scores·resolved config·
sanitized environment·manifest만 두고 전체 JSONL·이미지·열람본은 외부 root에 둔다.

---

## 8. 하드웨어·환경·재현

### 8.1 하드웨어 profile

| 항목 | rtx5060_8gb (로컬) | rtx4090_24gb / rtx3090_24gb (RunPod) |
|---|---|---|
| placement | cpu_offload (평가 smoke) / NF4 학습 | gpu_only |
| 용도 | 로컬 smoke·부분 평가·NF4 오답노트 RL | full 평가·승인된 학습 |
| gpu_weight_cap / reserve | 5.5 / 2.0 GiB | 20.0 / 3.0 GiB |
| min_free_gpu | 3.0 GiB | 12.0 GiB |
| disk offload | 금지 | 금지 |

- 로컬: 노트북 RTX 5060 8GB, RAM 24GB, WSL2. RAM과 VRAM은 합산 VRAM이 아니다.
  `device_map="auto"`는 OOM 방지/학습 성공 보장이 아니다. offload 뒤 `model.to("cuda")` 금지.
- full 900 실행은 RTX 3090 24GB에서 완료(4090 실측 아님). 다른 GPU 실행을 "4090 검증 완료"라 하지 않는다.
- allocator peak allocated/reserved가 물리 8GB보다 크면 완전 VRAM 상주라고 주장하지 않는다(paging 원인 미측정).
- 고가 GPU·다른 backend·양자화·greedy·batch 확대로 자동 전환하지 않는다.

### 8.2 환경

- `doctor`: GPU/freeVRAM·CPU/availableRAM·OS/WSL·driver·Python·torch/CUDA runtime·패키지·실제 CUDA/BF16 연산·
  disk/mount 기록. `nvidia-smi` 표시와 torch runtime을 구별한다. WSL에 Linux display driver를 설치하지 않는다.
- 같은 Python minor와 exact lock(`env/requirements-*.lock`). `latest`/무조건 `pip install -U` 금지.
- 확인된 성공 환경: Python 3.12.3, Torch 2.8.0+cu128, Transformers 4.57.1, vLLM 0.11.0.
- HF generate는 `use_model_defaults=False`로 학습 sampling을 모델 기본값이 덮어쓰지 않게 한다(2026-10-18 수정).

### 8.3 한 명령 재현

```bash
# 로컬 30문제 smoke (유효한 기존 실행은 반복하지 않는다)
bash scripts/eval.sh --protocol configs/eval/mmmu_val_v1.yaml \
  --hardware configs/hardware/rtx5060_8gb.yaml --model-ref manifests/models/baseline.json \
  --model-path "${MMDL_MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}" \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --run-id smoke-local-30 \
  --mode partial --subject Accounting --limit 30
# RunPod: 설치→doctor→취득→최소 GPU 검증→단일 900→검증/포장 (기본 dry-run)
bash scripts/reproduce.sh --stage evaluate --target runpod --suite mmmu-val \
  --commit "$MMDL_CODE_COMMIT" --protocol configs/eval/mmmu_val_v8.yaml \
  --hardware configs/hardware/rtx4090_24gb.yaml --model-ref manifests/models/baseline.json \
  --model-path "${MMDL_MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}" \
  --data-root "$MMDL_DATA_ROOT/evaluation/mmmu" --job-id "$MMDL_JOB_ID" --execute
# 저장 raw만 재채점
bash scripts/rescore.sh --artifact-root "$MMDL_ARTIFACT_ROOT" --run-id "$RUN_ID"
```

- 정확한 현재 인자는 `README.md`를 따른다. `reproduce.sh`는 학습/증강/final을 자동 시작하지 않는다.
- Bash는 `set -euo pipefail`, script 기준 root, 안전한 인자/실패 전파.
- 교수님은 RunPod 계정 없이 4090에서 `eval.sh` 한 명령으로 평가를 재현할 수 있어야 한다.
- 같은 코드/seed도 GPU 간 응답·점수 동일성을 보장하지 않는다. 절차/결과/재학습 재현을 구별한다. [W6]

---

## 9. manifest·재개·비용·회수·정리 gate

- 추적 사슬: `source revision → data/code/config → checkpoint → eval protocol → run → 성적표`.
- 모델 manifest: base/adapter/full/merged·base revision·artifact/hash·processor·train/data config·code commit.
  학습 재개에는 optimizer/scheduler/step/RNG/data 순서를 포함한다. LoRA 외 vision/projector(Merger)도 저장.
- 데이터 manifest: source/revision/split/license·부모 ID·변환/seed·상대경로/크기/hash·count·검수/오염검사.
- protocol hash는 P0/chat template/입력변환/dataset manifest/parser/evaluator를 묶는다.
  절대경로/run_id/시각은 execution manifest로 분리한다.
- 저장은 atomic/append-only. 재개 시 ID·모델/protocol/코드/환경/하드웨어 일치를 검사한다.
  원본 덮어쓰기·서로 다른 run 혼합 금지. 재채점 변경은 새 scoring version.
- 완료 상태는 독립적으로 기록한다: 구현/CPU 검사 · 로컬 SMOKE/PARTIAL · GitHub clean clone 재현 ·
  실제 GPU(doctor·BF16·device map) · FULL(900/900·raw·30과목·산술) · 로컬 회수(receipt) · 자원 정리(잔존 조회).
  미실행/실패를 PASS로 채우지 않는다.
- 회수: 닫힌 shard만 점진 회수, 최종 전체 hash 검사, 안전한 압축해제(경로 이탈/외부 symlink 금지),
  고유 ID/필드/산술·raw 재채점·이미지 열람 검증 → 로컬 receipt. 원격 exit code만으로 삭제하지 않는다.
- 회수 실패는 최대 3회 복구, 미검증 결과는 삭제하지 않고 GPU stop 후 `RECOVERY_REQUIRED/CLEANUP_PENDING`과
  storage 과금 가능성을 보고한다. 이미 발생한 비용을 0이라고 하지 않는다. API 잔액 snapshot ≠ 최종 청구서.

---

## 10. 코드·모델·데이터·결과 저장 위치

```bash
export HF_HOME="${HF_HOME:-$HOME/mmdl-cache/huggingface}"
export MMDL_DATA_ROOT="${MMDL_DATA_ROOT:-$HOME/mmdl-data}"
export MMDL_ARTIFACT_ROOT="${MMDL_ARTIFACT_ROOT:-$HOME/mmdl-artifacts}"
export MMDL_VENV_ROOT="${MMDL_VENV_ROOT:-$HOME/mmdl-envs/eval}"
```

- GitHub에는 코드·설정·lock·출처/manifest·작은 결과만. 대용량·venv는 WSL Linux 파일시스템.
- `MMDL_DATA_ROOT/evaluation/{mmmu,mmmu_pro}/`는 학습·증강 접근 금지. 학습 자료는 `training/` 아래.
- 로컬 디스크는 다운로드/변환/회수 peak를 고려해 **50GiB 여유**를 남긴다. 부족하면 임의 삭제하지 않는다.
- 유효 cache를 먼저 확인하고 재다운로드·실물 복제를 피한다.
- 주요 위치:
  - 오답노트 RL private 원장: `/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/`
    (V59: `verified-repair-notebook-v59/`, 그 이전 V1–V58 각 폴더).
  - 기존 학습 checkpoint: `/home/jang2/mmdl-artifacts/recovered/g6-full-*`, `tabmwp-{A,B}-2000-*`.
  - MMMU 평가 회수본: `analysis_exports/mmmu_20260923/`, `analysis_exports/mmmu_20260924/`.
  - 진단: `analysis_exports/bottleneck_20261003/`, `input_recovery_20261011/`, `g6_runtime_20261005/`.
  - 이 지침의 원본 백업: `analysis_exports/agents_md_backup_20261030/`.
- 강의 원본은 Git 제외 `local_sources/`에 두고 파일명/페이지/hash만 추적한다.
- Pro `standard (10 options)` test는 취득 시 전체 commit SHA를 고정하고 보관/hash 확인만 한다.

---

## 11. 1차 평가 실측 원장 요약 (R1 / R2)

| 항목 | R1 (2,048) | R2 (32,768 continuous) |
|---|---|---|
| run ID | `mmdl-val-20260922-3090-assignment` | `mmdl-cont32k-analysis-20260922-analysis` |
| commit | `af961a36…` | `702ed44c…` |
| 실행 | Transformers, batch1, SDPA MATH, deterministic | vLLM 0.11.0, 동시 2, non-deterministic |
| 정답(원래 채점) | 427/900 = 47.44% | 459/900 = 51.00% |
| 상한 종료 length | 226 (25.11%) | 75 (8.33%) |
| NO_PARSE | 96 | 69 (모두 객관식) |
| 생성 token 합 | 776,359 | 3,719,200 |
| 평가 시간 | 49,741.672초 | 57,885.846초 |

- 두 실행의 ID·질문·보기·정답·seed·이미지 순서/grid·입력 token 합 534,018 일치.
- paired: 둘 다 정답 347, 둘 다 오답 361, R1 오답→R2 정답 112, 반대 80. 순증 32(+3.56pp)는 길이·백엔드·코드·
  환경이 함께 바뀐 관측이며 순수 토큰 효과가 아니다.
- R2 동일 샘플 입력+출력 실측 최대 35,390 < 36,864. 75개 상한 종료는 문맥 부족 때문이 아니다.
- **파싱 결함 진단(D1)**: `parse_mcq_response`가 짧은 `LETTER. 보기 원문` 응답을 놓쳤다. R2 NO_PARSE 69 중 32개가
  형식 일치, 그중 24개가 정답(모두 stop 종료). R1도 31/23. → V4–V8 최종답 추출기로 해결.
- 공식 Qwen과 확인된 차이: 이미지 budget(우리 262144–1310720 vs 공식 1003520–4014080 → B2에서 공식값 채택),
  처리 경로, LLM 보조 채점·random fallback, seed(README 3407 vs 코드 42), 데이터(HF validation vs MMMU_DEV_VAL.tsv).
  각 차이의 점수 기여는 정량 입증하지 않았다.
- 원본: `analysis_exports/mmmu_20260923/originals/`(R1/R2 회수본, archive SHA는 원본 지침 §0.1),
  공유용 `upload/mmmu_analysis_text.zip`·`mmmu_images.zip`(959 PNG). Pod 2j4knzy36jtzmv·vf8wla9q5v9gm8 삭제 완료.

---

## 12. 이력 요약 (2026-09-21 → 2026-10-30)

상세 수치·SHA·PID·실패 원인은 원본 백업과 `docs/PROJECT_STATUS.md`를 본다. 아래는 결정과 결과만.

### 12.1 Baseline 평가 (09-21 ~ 09-28)

- 09-22: R1(2048) 427/900, R2(32K 연속 처리) 459/900 완료·회수·Pod 삭제.
  초기 5090은 호스트 UVM `EIO`, 초기 3090은 CUDA 12.7 vs cu128 부팅 조건 불일치 → 공식 base image digest로 해결.
  고정 batch2 → `LLMEngine.add_request()/step()` 연속 refill로 전환.
- 09-23: 감사 V2 재적용(536/584, 진단용), V4 채점기 확정(479/556). A/B/C 통제 각 6/6.
  동시1 C full900은 사용자 지시로 중단·부분결과 폐기. B2(동시2·공식 이미지 budget) 3090 실행.
- 09-24: B2 900/900 557(V4) → V5 561 → V6 → V7 REJECTED → **V8 확정 558/900**. MMDL 게시 `88dd56b`.
- 09-24~28: 보고서 템플릿 정리(`fa98910`), 격차 분석(`1b3e51b`), 기타 특이사항(`6b019df`),
  시간 표기(`197838c`, 최신). 보고서 경로를 `reports/mmmu_baseline.md`로 통일.
- 09-26~30: 내부 열람 HTML/PDF(`results/P1/`, 900/오답342, WebP 무손실로 512MB 미만),
  과목별 분석(`claudedocs/MMMU_V8_70pct_subject_analysis_20260930.md`).

### 12.2 병목 진단 (10-02 ~ 10-03, 비공개)

- Qwen 30과목/102문항(pilot60+coverage42), 816조건/749 실호출: 이미지 I 40/102, 텍스트 대체 O0 42/85,
  O1 31/69. 해설 전용 이미지 5장을 발견해 진단에서 제외(원래 결과 불변).
- Knowledge Oracle 15분야×2: K1(기초 지식) 30/30, K2(정확한 상태의 규칙 적용) 30/30. 전문 추론 전체 인증 아님.
- Gemma 동결 전수 감사(별도 protocol): 900 시도 중 유효 판정 미완료, 새 정확도 null. 원래 558/900 불변.
- 보고서 `claudedocs/Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md`, `..._Knowledge_Oracle_20261003.md`.
  Pod h447a4u6c0vq3k, re5qljwwyq7yui 삭제.

### 12.3 지도학습 시도 — 개선 실패 (10-04 ~ 10-12)

- 10-04: MMDL_test 저장소 정책 도입. G6(FL=LLM, FM=Merger, FV=Vision, FML=Merger+LLM) smoke를 A40에서 실행,
  기본 2배 예산 gate(필요 83,786초 > 잔여 48,418초)로 본학습 중단, Pod 삭제.
- 10-05~06: 3090에서 G6 full 10K×4조건(각 625 update). 동일 4K/P0/V8 **MMMU60: F0 36 → FL 22 / FM 27 / FV 21 / FML 23
  (모두 하락)**. 외부 dev: FM 25/48 > FML 21 > FL 18 = FV 18 > F0 15. 50K/최종900 자동 진행 없음.
  분석 `claudedocs/g6_full10k_final_analysis_20261005.md`. Pod 22u0uqi171v0cn 삭제.
- 10-11~12: 8K 입력 회복 진단 336호출(F0 48문항: I22/T24/IT26/IR20/TR22).
  TabMWP teacher-free A(답)/B(표 facts+답) 각 2000행/125 update: 외부 QA F0 35 / A 30 / B 28 (of 48),
  표 exact 19/11/17, 셀 268/253/299. 확대 gate 모두 false. Pod z2sa5xt3b93er9 삭제.
- **결론**: 지도학습은 MMMU·외부 QA를 개선하지 못했다 → 2026-10-16 사용자가 RL-only 오답노트 방식으로 전환.

### 12.4 오답노트 RL — 클라우드 시도 (10-16 ~ 10-18)

- 10-16: RL-only 설계(v1/v2) 작성, `fine_tuning/notebook_rl/` 신설.
- 10-17: A40 e10034q5chqqcr. 오염 검출기 후보 → 사용자 "후보면 중단" → 이후 "직접 대조 후 동일 문제만 blacklist"로 재개.
  native G9(OFF/동결 기울기 비교) 실패, vLLM 시작 메모리 비율 실패 → update 0, Pod 삭제.
- 10-18: 공유 HF 정책(24GB) 구현. 3090 kxxt6w3fpsb7ur는 `/dev/nvidia-uvm` EIO로 삭제.
  재시도 2uiqg6ym30ou3y: HF cold import guard 차단 수정, sampling 기본값 덮어쓰기 발견 → `use_model_defaults=False`,
  예산 한도로 update 0, Pod 삭제. 로컬 5060은 BF16 가중치만으로 VRAM 초과, native 경로 하한 26.5GiB 확인.

### 12.5 로컬 NF4 전환과 첫 실제 RL (10-21)

- 사용자 승인: 로컬 RTX 5060 8GB의 실제 Qwen4B **NF4/QLoRA** 자기수정 RL. 기존 NF4 cache 검증(68 logprob 차이 0).
- minimal-v3: **첫 실제 update1 + 별도 프로세스 fresh reload 통과**(L252/M24, V 동결 경로).
  노트 4개 보상 0. minimal-v4/v5: UNKNOWN 진단도 노트 생성(v2 protocol), Vision_ON update.
- 고정10 notebook arm: 실제 update2·90응답, both routes true; answer-only 대조군 update3·92응답.
  **외부 회로10: 첫 시도 F0/NFT/CFT 모두 5/10, 최종 모두 8/10 → 새 문항 개선 미관측.**
- 원본 BF16 두 blob을 복구 가능 격리 후 원본 경로 부재 상태 학습/reload 검증, 이후 복원.
- 전체 이력 오답노트 목표 시작(full-history-v1 → compact-v1 → schema-v2 → ranges-v3).
  전체 이력 v1/v2 update/reload 통과, 노트 보상 0.

### 12.6 전체 이력 노트 계약 반복 (10-22 ~ 10-25, V4 ~ V54)

- 노트 계약을 여러 번 바꿈: facts-first(V12), 독립 재관찰(V13), 부분 component 보상(V15), 사실 값 토큰 credit(V16),
  bounded synthesis(V25), 관계 부분보상(V29), OUTPUT FINALIZATION 관계(V31), 탐색 admission(V32/V33),
  공개 비교쌍·라벨 RL(V41–V43), 공개 참조형 전체 판정(V48–V54).
- 실제 update/reload 통과 사례: V15, V16, V17(고정10 5 update), V35, V43, V54 등. 그러나
  **strict 전체노트는 일관되게 0/N**(0/12, 0/16, 0/4 …), 학습 후 같은 이력 비교도 0/4→0/4.
- 검증기 결함도 발견·수정: 따옴표 관계값 파싱 오판(V43 고정10 중단), Trace 공백 문법(V26), 빈 인용 피드백(V35),
  native tree false-child 검출 누락(V17 전 TEN2 중단), checkpoint 교체 시 reload 증거 재사용(V59).

### 12.7 판정 단위 학습과 재설계 (10-26)

- V58: 판정 token 계수 정규화 → **signed assessment-only 기울기 비영 + update1 + fresh reload 통과**.
  기능 검증(F)이지만 strict 전체노트 0/4·오류→수정 관계 학습 0.
- 사용자 정정: 소규모 시험에서 성능 상승은 요구하지 않는다. 기능 검증과 성능 개선을 분리.
- **V59 재설계(verified-repair-notebook)**: T2/T3 한 completion의 ErrorCheck→RevisedSolution, bounded final(최대 3 claim),
  answer/online/final PG 분리, CORRECTED-only VJP gate, F/S/E 분리. 설계 3문서 + P04–P13 CPU 구현.

### 12.8 V59 실제 실행 (10-27 ~ 10-28)

- 10-27: 첫 고정2는 문법 오류 형제가 정상 형제의 T3를 막는 결함으로 중단 → repair-note-v2 수정.
  v6 최소시험 update1·저장·fresh reload: **F=true, S=false**. 노트 4개 `success_sample_id` 누락·claims=[].
  ASCII 빈줄 경계(v7), 탐색 admission 구현, frozen8 최소시험 F 통과.
  frozen8 고정10: 70풀이·24노트·update3 후 외부 14700초 한도(exit124) 종료. 노트 신호 0.
  optimizer moment CPU parking 구현(생성 중 약 1.077GiB 절약). 문법 실패 부모 재시도 v2.
- 10-28: 재관찰 v3 통합(115 CPU 검사), source10 최소시험 update1·fresh reload **F=true**.
  고정10 진행(update1→2→3, 모든 online/final VJP 0). claims-first 후보·실행기 v3 CPU 준비. → §4.

---

### 12.9 compact 노트와 데이터 정책 전환 (10-29 ~ 10-30)

- compact 노트 계약 `repair-note-compact-v1` 최소시험 v12: F + S_note 통과(update1, VISION_ON). 프롬프트 예시 gold 누설로 v11 무효·수정.
  고정10 compact·대조군 후 회로 평가10: F0·노트 FT·대조 FT 모두 엄격 0/10(바닥 효과, 개선 판단 불가).
- 사용자 정책 전환: 기존 데이터셋만·3,000문항 증명·11/17 마감·단일 답 원칙 → HF 풀 v3 구축·오염/품질 검사·동결(→ §4).

## 13. 반복 실패에서 얻은 교훈

1. **입력에 전체 이력을 넣는 것 ≠ 정확한 노트.** 실제 토큰 decode로 무손실 전달을 확인한 뒤에도
   모델은 성공 시도의 관측 오류를 연결하지 못하고, 올바른 재관찰 값도 버리고 기존 오류를 유지했다.
2. **동일 보상 = 상대 신호 0.** 네 후보가 모두 같은 점수면 노트 학습 기울기가 없다. 가짜 분산 금지.
3. **답 보상 update를 노트 학습으로 착각하지 않는다.** 대부분의 실제 가중치 변경은 T1 답 PG와 KL에서 왔다.
   online/final의 signed·positive VJP를 별도 채널로 확인한다.
4. **형식 통과로 내용 실패가 해결되지 않는다.** 숫자 문자열·키 이름을 고쳐도 관계·방향 오류가 남았다.
5. **검증기 결함과 모델 실패를 분리한다.** 따옴표·공백·빈 인용 등 표기 결함은 검증기를 수정하되
   기준을 완화하지 않고, 실제 내용 실패는 그대로 0점으로 둔다.
6. **같은 조건 GPU 재실행 반복 금지.** 원인을 바꾸지 않은 재시도는 비용만 쓴다.
7. **지도학습은 MMMU를 낮췄다.** G6 네 조건 모두 MMMU60 하락, TabMWP A/B도 외부 QA 하락.
8. **클라우드 실패는 호스트 단계에서 많이 발생했다**(UVM EIO, CUDA 부팅 조건, vLLM 시작 메모리 비율).
   모델 로드 전 doctor·실제 CUDA/BF16 검사를 먼저 한다.
9. **관찰자 스크립트 오류를 결과로 보고하지 않는다.** tuple/list 비교, 잘못된 파일명, cwd 혼동 등은
   원래 실패를 보존하고 관찰자만 정정한다.
10. **도구 세션 종료 ≠ 학습 종료.** 실제 PID/startticks·로그·result.json으로 판정한다.

---

## 14. 진행·제출·발표

`[수업: S3 pp.7,19–21; S1 §4; S8]`

- 팀장 public repo, 전체 팀원 collaborator, 각자 계정 commit/PR, default branch에 최종 병합.
- Assignment 1 마감 9/28 23:59 (완료). 본문 `reports/mmmu_baseline.md`, `assignment/assignment1.md`는 링크.
- PPP #1(모델 개선 계획안 발표): 9/29·9/30·10/6·10/7 팀 순서 배정, 최소 10분 + Q&A 5분,
  LMS `모델 개선 계획안 발표 - PPP #1`에 PDF 사전 제출(GitHub 제출과 별개).
- 발표 흐름: 평가 → 실패 사례 → 목표 → 관련 연구 → 데이터/학습법 → HW → 실험/일정.
  아직 실행하지 않은 목표를 성과로 쓰지 않는다. 47.44→51.00%는 실행 조건 묶음의 관측 변화,
  53.67%는 가정 계산, 학습 개선 성과는 아직 없음을 명시한다.
- 발표 자료(내부, 게시 안 함): `qwen3_mmmu_problem_improvement.pptx`(42장),
  `Qwen3VL_MMMU_발표_*.pptx` 등. 사용자가 저장한 원본을 먼저 대조·보존한 뒤 수정한다.

---

## 15. 근거 자료와 외부 출처

| ID | 자료 | 근거 |
|---|---|---|
| S1 | `local_sources/assignment_guidance.md` | 고정 스펙·채점·경로·마감 |
| S2 | `local_sources/SUBMISSION_TEMPLATE.md` | 보고 항목·실측·결과표 |
| S3 | `02_MMDL_Team_Project_Introduction.pdf` | 흐름·GitHub·평가·발표 |
| S4 | `03_MMDL_VLM_Introduction.pdf` | 입력/파싱·학습·데이터 금지 |
| S5 | `01_MMDL_Course_Introduction.pdf` | 범위·HW·Standard 10-way |
| S6 | `04_MMDL_Deep_Learning_1(2).pdf` | 파라미터·손실·최적화 (이론) |
| S7 | `05_MMDL_Deep_Learning_2.pdf` | 역전파·목적함수·softmax (이론) |
| S8 | 사용자 PPP#1 공지 | 발표 일정·시간·LMS |

사본은 이름이 아니라 hash로 확인한다. S6/S7은 이론이지 특정 학습률/배치/레이어 지시가 아니다.

| ID | 공식 근거 |
|---|---|
| W1 | Qwen recipe/processor: https://github.com/QwenLM/Qwen3-VL |
| W2 | 고정 model card: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md |
| W3 | Accelerate big model inference: https://huggingface.co/docs/accelerate/concept_guides/big_model_inference |
| W4 | CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html |
| W5 | Blackwell/PyTorch 2.7: https://pytorch.org/blog/pytorch-2-7/ |
| W6 | 재현 한계: https://docs.pytorch.org/docs/2.7/notes/randomness.html |
| W7 | MMMU 평가: https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu |
| W8 | Generation API: https://huggingface.co/docs/transformers/main_classes/text_generation |
| W9 | AGENTS.md: https://developers.openai.com/codex/guides/agents-md |
| W10 | RunPod Stop/Terminate: https://docs.runpod.io/pods/manage-pods |
| W11 | RunPod storage: https://docs.runpod.io/pods/storage/types |
| W12 | RunPod pricing: https://docs.runpod.io/pods/pricing |

Qwen 평가 코드 대조 commit `96588727e44c78b25ba03ea03b8e12f7e64fd0da`(`run_mmmu.py`, `infer_instruct.sh`,
`eval_utils.py`, `dataset_utils.py`), MMMU evaluator commit `268471d0d488258990025331c7528359c324aa25`.
상세는 `docs/SOURCE_REVIEW.md`. 모델 card VL 길이 16384와 평가 recipe 32768은 출처가 다르다.

주요 내부 문서:
- `docs/PROJECT_STATUS.md` — 상태·결정·증거 원장(최신 정본)
- `docs/EVALUATION_V2.md` — 채점 버전 조건
- `fine_tuning/README.md`, `fine_tuning/notebook_rl/README.md` — 학습 코드 사용법·현재 상태
- `claudedocs/notebook_rl_redesign_20261026/` — 오답노트 RL 설계 정본
- `claudedocs/research_error_notebook_self_correction_methods_20261026.md` — 관련 연구
- `claudedocs/g6_full10k_final_analysis_20261005.md` — 지도학습(G6) 결과
- `claudedocs/Qwen3_VL_4B_Bottleneck_Diagnostic_Report_20261003.md` — 병목 진단

---

## 16. 이 문서의 유지 규칙

- `AGENTS.md`와 `CLAUDE.md`는 **항상 byte 동일**하게 유지한다. 한쪽을 고치면 다른 쪽에 복사하고 `cmp`로 확인한다.
- 이 문서는 규칙·현재 상태·요약 이력만 담는다. 목표 분량 약 800줄. 날짜별 상세 원장(수치·SHA·PID·실패)은
  `docs/PROJECT_STATUS.md`와 private 원장에 기록하고, 여기에는 §4 현재 상태와 §12 이력에 1–3줄만 추가한다.
- §4를 갱신할 때 이전 §4 내용은 §12의 해당 날짜 항목으로 1–3줄 요약해 옮긴다.
- 과거 기록을 소급 수정하지 않는다. 잘못된 서술은 새 항목으로 정정하고 원래 기록은 원본 백업에 남는다.
- 규칙(§2)을 바꾸는 것은 사용자의 명시적 지시가 있을 때만이다. 변경 시 날짜와 지시 내용을 함께 적는다.
- Codex 지침 로드 한도(`~/.codex/config.toml`의 `project_doc_max_bytes`, 현재 262144)를 넘지 않게 한다.
