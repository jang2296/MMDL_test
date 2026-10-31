# 오답노트형 자기개선 RL — Codex 작업 감사, 로컬 파이프라인 보완, 대규모 학습 데이터·RunPod 계획

작성 2026-10-30 · 대상 Qwen/Qwen3-VL-4B-Instruct@ebb281ec · 내부 문서(게시 금지)

> 상태 표기: `[실측]` 저장 근거 확인 · `[CPU]` CPU 검사만 · `[계획]` 미실행 · `[판단]` 근거 기반 추론.
> 이 문서는 대규모 학습 승인 문서가 아니다. RunPod·유료 실행은 사용자 승인 후에만 한다.
> 사용자 요청문에 "지도 학습 파이프라인"이라는 표현이 있었으나 프로젝트 규칙상 SFT는 금지이므로,
> 이 문서의 "학습"은 모두 **독립 검증 보상 기반 RL**을 뜻한다.

## 1. 요약 (Executive summary)

<!-- RESULTS_SUMMARY -->

## 2. Codex(V59) 작업 감사 결과 `[실측]`

근거: 비공개 `verified-repair-notebook-v59/` 원장, `fine_tuning/notebook_rl/` 코드 대조(읽기 전용 감사).

| 단계 | 구현 위치 | 실제 모델 근거 |
|---|---|---|
| T1 풀이 n4 | `rollout.build_tree` | 있음 (ten-functional-v10 raw) |
| 정오 피드백(정답 미노출) | `rollout.repair_parent_feedback` | 있음 |
| 이전 풀이 포함 재시도 T2/T3 (ErrorCheck→RevisedSolution) | `rollout.continuation_ids`, `repair_contract.parse_retry_sections` | 있음, 30개 중 11개 문법 실패(UNCERTAIN_REFS 등) |
| 전체이력 최종 노트 n4 | `repair_notebook.append` | 20개 생성, **20개 모두 claims=[]**, 14개 success_sample_id 누락으로 스키마 실패 |
| 노트 검증 | `repair_credit.audit_final_note` | 검증할 주장 0 |
| RL 업데이트 | `repair_training`→`updates.optimizer_step` | 3회 커밋, **신호는 답 보상뿐**(PG 토큰 828/2045/1462) |
| Vision 경로 | `episode_notebook.strict_routing` | V59에서 **VISION_ON만** 관측, OFF-only update 없음 |
| 저장·새 프로세스 재로딩 | `checkpoint.save`, `local_train reload` | 최소시험만 통과(F=true), 고정10은 재로딩 없음 |

핵심 원인:
1. **노트 계약이 4B 모델에 비현실적**이었다. 입력은 12.5k토큰의 정렬·압축 JSON(escape된 원문, 공개 origin 사전,
   피드백 wire 템플릿), 출력은 긴 hex sample_id 복사·attempts 전수 목록·`:rN` 정규 참조를 요구했다.
   하나라도 틀리면 노트 전체가 0점이라 네 노트가 같은 0점 → 그룹 분산 0 → 노트 기울기 0.
2. **S 정의가 CORRECTED(이후 시도에서 실제로 고쳐진 오류) 전용**이었다. 감사 가능한 이력에서 그런 교정 기회 자체가 0이었다
   (재시도가 관측 오독을 고치지 않고 정답만 회복). 형식을 고쳐도 S가 생길 수 없는 구조였다.
3. 고정10은 두 번(source8/10) 모두 업데이트 3회 뒤 외부 timeout으로 끝났다. 1회 업데이트 이후 backward가 10–20배
   느려졌다(AdamW moment 1.16GB가 GPU에 남아 8GB를 넘김 → 공유메모리 paging, 추정).

## 3. 데이터·오염·규칙 감사 `[실측]`

- 현재 학습 10문항: 프로그램 생성 그래프(BFS/DFA/트리/최단경로), 정답 프로그램 계산(`generators/graph.py`),
  `benchmark_derived=false`. 평가 10문항: 회로 ladder 3위상, 학습과 출처·생성기·구조·이미지 hash 분리 검증.
- MMMU/MMMU-Pro 제외 인덱스: 100 Parquet/16,740행(MMMU dev150·val900·test10500, Pro standard10/4·vision 각1730),
  검출기 exact/n-gram/pHash·dHash/SSCD/SigLIP/OCR. 학습 풀 40문항 전체 12단계 스캔 완료. 후보 472건(고정10)은
  AI 검토로 DISTINCT(사람 검증 아님).
- 규칙: notebook_rl에 SFT/CE 경로 없음, 상위모델 import 차단, ActorItem에 gold 필드 없음.
- 보완 필요: ① 학습 측 OCR이 렌더 이미지를 읽지 않음(주석 사용), ② 회로 평가셋은 오염 스캔 미실시(평가용이라 무해,
  학습 재사용 시 필요), ③ actor guard가 로컬 MMMU HF 캐시 경로를 금지하지 않음 → 이번에 추가,
  ④ 로컬 러너에 CE 런타임 차단 없음(코드 경로는 없음).
- **구조적 한계:** 관측 슬롯 정답이 없는 일반 MCQ/수치 데이터는 경로가 UNRESOLVED → 학습에서 통째로 제외된다
  (답 보상도 미사용). 대규모 다분야 학습 전에 "슬롯 없는 데이터용 명시적 경로"가 필요하다(§6).

## 4. 이번 보완: model-friendly compact 오답노트 (`repair-note-compact-v1`)

**설계 원칙:** 검증기·보상 규칙·경로 판정은 그대로 두고, 모델이 쓰는 형식만 4B 모델이 실제로 쓸 수 있는 형태로 바꾼다.
host는 "시도 번호·사실 이름 → 기존 감사 결과" 대응(부기)만 하고 판정은 하지 않는다.

- 입력(`compact_note.render`): 문제·보기, 모든 시도의 원문 그대로(번호 1..N, "first try"/"retry of attempt k"),
  각 시도의 공개 정오는 **CORRECT / NOT CORRECT / NOT GRADED**만 표시(상태 코드가 진단을 대신 알려주지 않게),
  첫 재시도 전 미검증 재관찰 원문, 원본 이미지. 정답·gold·오류 목록 없음.
- 출력: `{"errors":[…최대4 채점], "vision_misread":"YES|NO|UNSURE", "lesson":"…"}`
  - `OBSERVATION{attempt,item,correct}`: 그 시도의 해당 사실이 fresh native 감사에서 VIOLATED이고 제안 값이 private
    verifier의 1-사실 관측 검사에서 COMPLIANT면 +1, 사실이 원래 맞았으면 −1(거짓 지적), 위치만 맞고 값이 틀리면 0.
  - `NO_FINAL_ANSWER{attempt}`: 실제 결과가 최종답 없음/잘림이면 +1, 답이 있었으면 −1.
  - `WRONG_STEP{attempt}`: 그 시도의 TRACE 검사에 VIOLATED가 있으면 +1, 전부 COMPLIANT면 −1.
  - `vision_misread`: 독립 `strict_routing` 판정과 일치(YES↔VISION_ON, NO↔VISION_OFF) +1, 반대 −1, UNSURE/UNRESOLVED 0.
    **실제 학습 경로는 이 자기판단이 아니라 독립 판정으로 정한다.** 자기판단은 검증·보상 대상일 뿐이다.
  - 노트 보상 = (항목 점수 합 + vision 점수)/5 ∈ [−1,1]; lesson은 기록만 하고 보상·credit 제외.
- 학습 연결: 원래 n4 노트 그룹 평균으로 중심화한 advantage를 lesson을 뺀 노트 토큰에 final 채널 계수로 적용
  (기존 cap 8192·loss 식 유지). 검증 통과 항목(또는 vision 정답)이 있는 양의 advantage 노트 토큰을
  `positive`로 집계해 **S_note(verified-note-claim-v1)**로 보고한다. 기존 CORRECTED 전용 S와 합치지 않는다.

**실제 무업데이트 probe 결과 `[실측]`** (F0 NF4, 저장된 실제 이력 6개, n4, 노트 1024토큰, optimizer 0,
파라미터·NF4 기저 불변 확인; `compact-note-v1/gpu-probe-v3/`, 재채점 `regrade-v2.json`)

| 지표 | v2 계약(고정10 실측) | compact(probe) |
|---|---|---|
| 노트 파싱 | 6/20 | **24/24** |
| claim 있는 노트 | 0 | 24 |
| 검증 통과 진단 | 0 | **8** |
| 반박(거짓 지적) | – | 16 |
| vision 자기판단 정답 | – | 12/24 |
| 보상 분산이 있는 이력 | 0/5 | **4/6** |

예: 이진트리 이력(raw-0056)에서 노트들이 "시도1의 child[F,L]은 틀렸고 그림 값은 A" 같은 실제 오독과 정답값을
지목했고 verifier가 확인했다(보상 [0.8, 0.8, 0.4, 0.0]). 반면 최단경로 이력에서는 모델이 e8 방향을 일관되게 반대로
보아 거짓 지적이 많았다 — 노트도 같은 시각 오류를 공유한다는 뜻이며, 검증 보상이 바로 이 부분을 벌점으로 교정한다.
한계: 6개 이력·F0 정책의 진단이며 학습 효과가 아니다. vision 자기판단은 현재 데이터가 거의 VISION_ON이라
"항상 YES"로 쏠릴 위험이 있다(OFF 이력 확보 필요).

## 5. 로컬 파이프라인 검증 결과

모두 실제 사전학습 Qwen3-VL-4B-Instruct NF4(기저 동결, Vision/LLM LoRA + Merger4 학습), 로컬 RTX 5060 8GB.
증거 루트 `/home/jang2/mmdl-artifacts/notebook-rl-local-nf4-20261021/verified-repair-notebook-v59/`.

**5.1 단계별 결과 `[실측]`**

| 단계 | 결과 |
|---|---|
| 코드 | compact 계약 opt-in 통합(기본 v2 hash `89181dfc…` 불변), 전체 시험 변경 전후 동일 기존 실패41·신규 실패0, 핵심 129·84검사 PASS, CUDA parking 13 PASS |
| 독립 검토 | gold 누설 차단 결함1(프롬프트 예시 = 학습 문항 정답) 발견·수정·회귀시험, v11 무효 보존 |
| 데이터·캐시 | v12 동결 395파일, 고정10 동일·차단0, NF4 기저 불변·원본 접근0 |
| 최소시험(2문항) | update1, **S_note=true**(양성 검증 토큰161), 변경 L252/M24/V48, 624 실물 PASS, **새 프로세스 재로딩 PASS** |
| 고정10 | 6215초 정상 종료(이전 두 번은 시간초과), update4, **S_note=true 2회**(토큰1154·268), 2회 무신호(노트 분산0), 마지막 쌍 UNRESOLVED(정답 미도달), 624 실물 PASS(step4), **재로딩 PASS**(S_gradient_verified·repair_learning_verified) |
| 경로 | 모든 실제 업데이트가 V 포함 → **VISION_ON만 실측**, VISION_OFF-only 미관측 |
| 성능 | backward/train 75–267초(v10 525–3180초), peak allocated 8.46GB·reserved 10.66GB(8GB 물리 상주 아님) |

**5.2 파이프라인 단계가 실제로 작동했는지 (사용자 요구 대조)**
1. 풀이 → 정오 피드백(정답 미노출) → 이전 풀이 포함 재시도 → 정답 도달: 작동(T1 n4, T2/T3 n2, 최대3단계).
2. 정답 후 전체 이력을 보고 모델이 직접 오답노트 작성: 작동(compact 노트, 성공 풀이 포함 전 시도 입력).
3. 노트 내용(무엇이 틀렸고 그림 값이 무엇인지, 비전 오독인지)이 독립 검증되어 보상으로 학습: **작동**(S_note, 검증 토큰의 VJP 비영, 실제 가중치 변경·저장·재로딩).
4. 비전/Merger+LLM 경로 결정: 노트의 자기판단이 아니라 독립 관측 검증으로 결정, 노트 판단은 채점만. ON 경로 실측, OFF 경로는 이번 데이터에서 발생하지 않아 미검증(이전 프로토콜 2026-10-21에서 OFF-only update 실측 기록은 있음).
5. 노트 보상 분산이 없는 그룹은 학습 신호 없음으로 그대로 기록(조작 없음).

**5.3 주의:** `result.notebook_signal_groups`는 legacy 스칼라 노트 보상 집계라 repair 계약에서는 항상 0이다.
노트 학습 근거는 `repair_aggregate_gradient`(S, 양성 검증 토큰, 채널별 VJP)다.

### 5.4 별도 회로 평가 10문항: 학습 전 / 노트 학습 / 정답 보상 대조군 `[실측]`

같은 동결 평가 10문항(native 회로 sealed, 학습 문항과 출처·구조 분리), n1 첫 시도 → 검증기 피드백 재시도 → compact 노트,
평가 중 학습 0. 세 arm 모두 같은 NF4 베이스·같은 seed 식. 원본: `verified-repair-notebook-v59/eval-{F0-notebook,FT-notebook,FT-answer-only}-compact-v12/`
(각 CPU replay `CPU_NATIVE_LADDER_RECOVERY_REPLAYED`). 표 생성 스크립트는 scratchpad `evalcmp/compare.py`(내부).

| arm | 학습 update | 첫 시도(엄격) | 첫 시도(관대: 정답 값이 응답 어딘가에) | 최종(재시도 후, 엄격) | 회복 | 길이 한도 종료/호출 | 생성 시간 | 생성 토큰 | 노트 | 평가 peak alloc/reserved |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 (학습 전) | 0 | 0/10 | 1/10 | 0/10 | 0 | 11/30 | 4,904초 | 26,590 | 0 | 5.62/6.67 GiB |
| FT 노트 학습 | 4 | 0/10 | 1/10 | 0/10 | 0 | 8/30 | 5,128초 | 25,237 | 0 | 5.63/6.70 GiB |
| FT 정답 보상 대조군 | 3 | 0/10 | 1/10 | 0/10 | 0 | 16/30 | 5,510초 | 27,509 | 0 | 5.64/7.33 GiB |

- 관대 채점 1/10은 세 arm 모두 같은 문항(000011)이다. 이 평가 문항은 모두 단위가 붙은 분수 정답(주관식)이라
  Qwen 방식의 무작위 찍기 점수는 해당하지 않는다.
- **바닥 효과**: 세 arm 모두 엄격 0/10이라 노트 학습과 대조군의 차이를 판별할 수 없다. 길이 한도 종료 수 차이(8 vs 16)는
  단일 seed·10문항이라 개선 주장 근거가 아니다. 재시도 성공이 없으니 노트도 0개다.
- 학습 쪽 사실: 노트 arm 고정10은 4 update 중 2개(pair 1·2)에서 `S_gradient_verified=true`(검증된 노트 항목의 양성 VJP 비영),
  대조군은 3 update 모두 S=false(노트 채널 없음). 학습 peak allocated 8.46/8.99 GiB는 물리 8GB를 넘는 allocator 수치이며 완전 VRAM 상주를 주장하지 않는다.
  두 arm 모두 624텐서 실물검사·새 프로세스 재로딩 PASS(별도 receipt).

### 5.5 본 학습 보상 구성 결정 (2026-10-30, 잠정) `[설계]`

- **결정**: 본 학습 기본값은 **정답 보상 + 최종답 보상 채널**(정답 중심)으로 한다. 노트 보상 채널은 300문항 파일럿에서
  사전 등록한 **비교 arm**으로만 넣고, 파일럿의 과목별 점검 문항(풀 dev)에서 여러 seed 같은 문항 짝 비교로 정답+최종답 arm보다
  첫 시도 정확도 또는 재시도 회복률이 나아지고 과목별 하락이 없을 때만 본 학습에 포함한다.
- **근거**: (1) 노트 채널은 기능적으로 동작한다(F 통과, S_note 2/4 update). (2) 그러나 새 문항 효과(E)는 위 표처럼 판별 불가이고,
  2026-10-21 회로 평가에서도 F0/노트/대조 첫 시도가 모두 5/10이었다 — 노트의 추가 효과는 두 번 모두 관측되지 않았다.
  (3) 노트 생성은 해결된 문항마다 n4×최대 8192토큰을 더 쓰며, 관측 슬롯이 없는 일반 문항에서는 텍스트로 확인되는 항목(최종답 누락·풀이 단계)만
  보상할 수 있어 신호가 약하다. (4) MMMU 오답의 25%가 길이 한도 잘림·6%가 최종답 누락이라 최종답 보상은 직접적인 손실 원인을 겨냥한다.
- 이 결정은 로컬 풀 기반 고정10·평가(바닥 효과가 없는 dev 문항)를 다시 측정한 뒤 갱신한다.


## 6. 대규모 학습 데이터 계획 `[계획]`

### 6.1 현실적 목표 `[판단]`
- 공개 RL-only 결과(3B–7B Qwen VL)의 MMMU 변화는 −0.7~+4.9pp, 중앙값 약 +2pp
  (VL-Rethinker 54.3→56.7, ThinkLite 50.6→55.5, SophiaVL 순수 GRPO 58.7→58.0, Continual-RFT 52.1→54.2;
  같은 연구에서 SFT는 52.1→40.1로 급락).
- 우리 엄격 V8 기준 558/900(62.0%). 67.4 초과에는 +5.4pp가 필요해 소규모 RL만으로는 가능성이 낮다.
  격차 일부는 평가 절차(Qwen은 규칙 추출 실패 시 GPT judge 사용, 데이터 MMMU_DEV_VAL.tsv)다.
  우리 저장 응답만으로도 추출 정책 차이로 556→584가 됐다. **현실 목표: 엄격 기준 +1.5~3pp(63.5~65%)**,
  공식식 채점 수치는 별도 병기한다. 900문항 표준오차 약 1.6pp.

### 6.1b 62 → 67.4 격차 실측 분해 `[실측+판단]` (2026-10-30, 저장 B2 응답 900개 재채점, 새 추론 없음)

근거: `analysis_exports/feasibility_67_20261030/`(스크립트·JSON·워크플로 원장). 검증자가 핵심 수치를 원본에서 독립 재계산·일치 확인.

| 채점 방식 | 같은 B2 응답 점수 |
|---|---|
| 우리 엄격 V8 | 558/900 = 62.00% |
| 공식 MMMU 평가기(마지막 언급 보기 추출, 실패 시 무작위) | 결정적 445(49.4%), 기대값 463.2(51.5%) — 오히려 더 낮음 |
| Qwen 방식(규칙 추출 398개 + GPT 판정 502개를 모델링, 실패 시 무작위) | 기대 583.3(64.8%), 판정 문맥초과 가정 579.8(64.4%), 낙관 586.1(65.1%) |

- Qwen식 채점이 엄격 기준보다 약 +2.4~3.1pp 높지만 대부분 **잘린 응답의 무작위 찍기 점수(+2.14pp)와 주관식 정답 문자열 관대 일치(+1.33pp)**이며, 실제 답을 더 잘 읽어서가 아니다(판정기는 실행하지 않고 모델링).
- 오답 342개 구성: 명시적 오답 236, 출력 상한(32768) 잘림 85, 자연종료 추출 실패 21. 잘림 90개 중 V8 정답 5개뿐.
- 다른 실행(R2)에서 맞힌 오답은 57/342뿐, 271문항은 세 실행 모두 오답 — 지식·추론이 주 병목, 이미지 지각은 일부.
- **엄격 기준 67.4 초과 = 607/900 이상 = +49문항의 실제 능력 향상 필요.** 현재 합성 10문항 학습만으로는 전이 불가(직접 대응 문항 27개, 다 고쳐도 +1.9pp). 확장 계획(다분야 비-MMMU 혼합 + 최종답/반복생성 방지 보상)으로도 검증자 판단 확률 약 5%(2–10%), 현실 도달 63–65%. Qwen식 채점 기준은 확장 계획 시 약 15–20%.
- 가장 큰 모델 측 손실 풀: 반복·잘림 응답 90개 → 최종답/반복 방지 보상이 1순위 개선 수단(정당한 검증 보상).

### 6.2 데이터 원칙
1. 라벨은 사람 작성 또는 프로그램 생성만. 모델이 만든 질문·정답(PMC-VQA, ArxivQA, SciGraphQA, MathV360K 증강분,
   VisualWebInstruct-Verified)은 제외. 벤치마크(We-Math, MedXpertQA)는 제외.
2. 모든 후보는 기존 제외 파이프라인(텍스트 n-gram + pHash/SSCD/SigLIP + OCR)으로 MMMU/Pro 전 split과 대조,
   후보는 직접 대조 후 동일 문제만 family째 blacklist. 교과서 출처(TQA, ScienceQA, PathVQA, MMK12)는 특히 주의.
3. 난이도 필터: F0의 n8 통과율 1/8~7/8만 사용(전부 맞거나 전부 틀린 문항은 그룹 분산 0 → 기울기 0).
4. 분야별 non-MMMU dev 게이트(출처별 약 100문항 보류). MMMU val은 최종 측정에만, early stopping에 사용 금지.
5. 망각 방지: KL(참조 대비) 유지(현재 β=.01; 문헌 1e-3 수준도 비교), LoRA 유지, 분야 가중 샘플링, 각 업데이트 후
   분야 dev 회귀 감시, 회귀 시 해당 체크포인트 폐기.

### 6.3 제안 혼합 (원 데이터 약 24k → 필터 후 6–12k 예상)

| MMMU 분야(검증 문항) | 후보(원 데이터) | 라벨 | 비고 |
|---|---|---|---|
| Tech & Eng (210) | Geometry3K 2.0k · GeoQA 2.0k · 자체 회로/그래프 생성기 2.0k | 주석/프로그램 | 회로는 관측 슬롯 보유 → 오답노트 전체 경로 |
| Science (150) | ScienceQA(자연과학·이미지) 1.5k · AI2D train 1.5k · TQA 도식 1.5k · MMK12 1.5k | 원 출처 | AI2D는 원본 train split만 |
| Business (150) | TabMWP 2.0k · ChartQA 사람작성분 1.5k · PlotQA/DVQA/FigureQA 1.5k | 원 출처/템플릿 | ChartQA 기계생성 23.1k 제외 |
| Health & Medicine (150) | SLAKE-en 1.0k · VQA-RAD 0.8k · PathVQA(closed) 1.2k | 의사/검수 | 라이선스 확인 필요 |
| Humanities & Social (120) | ScienceQA 사회과학·이미지 1.2k · MapQA 0.8k | 원 출처/템플릿 | 역사·문학·심리 적합 RLVR 세트 없음 |
| Art & Design (120) | SSMR 악보 1.0k · WikiArt 메타데이터 MCQ 1.0k | 프로그램 | 비상업 연구 한정 |

빈 영역(역사·문학·심리·약학)은 학습하지 않는다. 억지로 채우려 모델 생성 데이터를 쓰지 않는다.
대신 해당 분야 dev 회귀만 감시한다.

### 6.4 파이프라인 적용 방식
- **관측 슬롯이 있는 프로그램 생성 데이터**(그래프·회로·차트·화학·정역학·악보 생성기): 오답노트 전체 경로
  (재시도→compact 노트→독립 경로 판정→노트+답 보상).
- **슬롯 없는 일반 데이터**(ScienceQA 등): 새 명시 프로토콜 "answer-verified route"가 필요 — 답 보상 + 재시도,
  노트는 생성·기록하되 검증 가능한 항목(NO_FINAL_ANSWER 등)만 보상, 경로는 사전 고정(기본 Vision 동결·Merger+LLM,
  관측 근거가 없으므로 Vision 학습 근거 부족). 현재는 UNRESOLVED로 버려지므로 **미구현**이며, 대규모 전 구현 대상.
- 생성기 확장은 `source_registry_v1`의 gen-chart/music/chem/statics(관측 슬롯 보유)를 프로토콜 허용 목록에 추가해야 한다.

## 7. RunPod 대규모 학습 준비 `[계획]`

사용자 승인 전에는 아무 Pod도 만들지 않는다. 아래는 승인 즉시 실행할 수 있게 고정한 순서와 조건이다.

**7.1 단계 (각 단계 통과 후 다음 단계, 실패 시 원인 보존·중단)**
1. 코드 고정: 로컬 검증된 `fine_tuning/`을 `scripts.sync_test_repo`로 MMDL_test feature branch에 정상 게시,
   전체 commit SHA 기록. Pod는 그 SHA를 clean clone한다(로컬 코드·가중치·데이터 업로드 금지).
2. 환경: 기존 `fine_tuning/runpod_setup.sh` 계열의 exact lock 설치 → `pip check` → CPU 회귀 → 실제 CUDA/BF16 검사 →
   지정 모델 13파일 hash. (과거 실패: UVM EIO, CUDA12.7 부팅 조건, vLLM 시작 비율 — 모델 로드 전 doctor로 거른다.)
3. **BF16 소규모 검증(별도 승인)**: 같은 오답노트 파이프라인을 24GB+ GPU에서 최소 2문항·업데이트1·저장·새 프로세스
   재로딩으로 확인. 현재 `local_train`은 NF4 정책만 로드하므로 BF16 정책 로더 연결(`policy.build_policy`)이
   선행 구현 항목이다(§8). NF4 경로를 그대로 24GB GPU에서 돌리는 것은 코드 변경 없이 가능하다.
4. **대규모 학습(별도 승인)**: §6 데이터의 오염 검사·통과율 필터를 끝낸 고정 manifest로만 실행.
   분야별 dev gate 회귀 시 중단, MMMU val은 최종 1회 측정.

**7.2 GPU 선택 `[판단]`**
| GPU | VRAM | 용도 | 비고 |
|---|---|---|---|
| RTX 3090/4090 | 24GB | NF4 경로 그대로, 또는 BF16+LoRA 소규모 | 로컬 8GB의 공유메모리 paging 없음 → backward 대폭 단축 예상(미측정) |
| A40 | 48GB | BF16+LoRA+Merger 대규모, 긴 이력(24k 문맥) | 과거 Pod에서 G9/vLLM 시작 비율 실패 이력, 수정본 미검증 |
| ("A50") | — | RunPod 표준 목록에 없음. A40(48GB) 또는 RTX A5000(24GB)을 뜻한 것으로 보고 위 두 등급에 포함 | |

**7.3 비용·시간 산정 방법** — 로컬 실측 문항당 시간(§5)을 GPU 등급 배율로 단순 추정하지 않는다.
첫 Pod에서 200문항 pilot의 실제 생성·backward 시간을 측정하고, 2배 여유 예산 gate를 통과할 때만 본 실행.
watchdog: 최초 과금 시각 고정, GPU/금융 종료 시각, 회수 예약금, 10분 점검, 결과 로컬 회수·검증 후 소유 Pod 삭제.

### 7.4 허깅페이스 학습 데이터 선정 `[조사+독립 검증, 2026-10-30]`

과목군별 조사 에이전트 3명과 반박 검증 에이전트 3명이 각 카드/README·datasets-server를 직접 확인했다.
근거: `analysis_exports/runpod_data_speed_plan_20261030/`(journal_results.json, 카드 사본, 검증 노트).
모든 중복 위험 등급은 판단이며 **로컬 오염 검사(기존 검출기, MMMU dev/val/test + MMMU-Pro 전 split) 전에는 사용 금지**.
the_cauldron은 MMMU **test** 이미지만 제거했으므로 그 계열도 직접 검사한다.

| 과목군 | 포함(include) | 조건부(conditional) | 제외와 이유 |
|---|---|---|---|
| 과학·공학 (Biology, Chemistry, Physics, Geography, Math, CS, Agriculture 등) | ScienceQA train 이미지 175, AI2D 75, TQA 70, Geometry3K 50, MapQA 30, PlantVillage 70, DeepWeeds 30, GVLQA 70 | PGPS9K 20(신청서 필요), GeomVerse 20, MAVIS-Function 30(규칙 엔진 행만), ChartQA 사람 질문 20, IP102 20, CGHD 회로 40, CubiCasa5K 40, COD 결정구조 60, PubChem SMILES 50 | MMK12(LLM 번역), Geo170K 계열(LLM 보강), ViRL39K(재작성·집계), Zebra-CoT(모델 풀이), CircuitSense·mechanical-reasoning(벤치마크), IconQA·CLEVR-Math(관련성 낮음) |
| 경영·인문사회 (Accounting, Finance, Economics, Manage, Marketing, Sociology, History, Literature, Psychology) | TAT-DQA 100, FinQA 70, TAT-QA 15, ChartQA 사람 질문 155, PlotQA 45, HiTab 40, ScienceQA 사회·언어 85, TabMWP 55 | MapQA 원본 40, MultiHiertt 20, InfographicVQA 35, VCR 20, New Yorker 20, OCR-VQA 20, EXAMS-V 10 | ChartQA 기계 생성 질문, SlideVQA, DVQA·FigureQA, DocVQA(평가 split), plot_qa |
| 의학·예술 (Basic/Clinical Med, Diagnostics, Pharmacy, Public Health, Art, Art Theory, Design, Music) | VQA-RAD 40, SLAKE 영어 50, NCT-CRC-HE 30, 골수 세포 55, PTB-XL ECG 65, ChEBI-20-MM 120(RDKit 렌더링), Met Open Access 90, Cooper Hewitt 50, Crello 30, SSMR-Bench Train_visual 100 | WikiArt 70(이미지 유사도 검사 필수), GrandStaff 20, PathVQA 예비 | PMC-VQA(ChatGPT 생성), OmniMedVQA·MedXpertQA(평가용), SemArt, PrIMuS, MusicTheoryBench, MusicScore |

- 합계: 공개 데이터 원시 문항 약 2,300개(포함 약 1,765 + 조건부 약 555). 3,000개를 채우려면 약 700개를
  **프로그램 생성기**로 만들어야 한다(정답을 프로그램이 계산; 상위 모델 생성 금지).
- **공개 데이터가 없는 과목(판단)**: Energy_and_Power, Mechanical_Engineering, Materials(일부), Electronics 회로 해석,
  Architecture(구조), Public_Health, Manage(CPM/PERT 등), Psychology, Literature, History(대학 수준).
  → 기존 `gen-circuit-v1`·그래프 생성기 확장 또는 감시만 한다.
- 수준 한계: 대부분 초중등·분류 수준이라 MMMU(대학 수준) 전이는 제한적이다(§6.1b의 63–65% 판단과 일치).
- 데이터 정답 형식 주의: GVLQA 최단경로·위상정렬은 정답이 여러 개일 수 있어 그래프로 검사하는 verifier 필요,
  Geometry3K 정답 일부가 LaTeX 식이라 기호/허용오차 비교 필요, MapQA 일부는 주(州) 목록이라 집합 비교 필요.
- 대부분 관측 슬롯이 없으므로 현재 `strict_routing`에서 UNRESOLVED로 버려진다 → §7.5의 **정답 검증 경로** 필요.

### 7.5 속도·비용 계획 `[코드 점검 실측 + 판단]`

**병목은 GPU가 아니라 생성 코드다.** `HFRolloutEngine.sample_with_settings`가 요청·seed마다 `model.generate`를 따로 호출해
항상 한 시퀀스씩 디코딩한다. 2026-10-18 RunPod 3090 BF16 실측 13.7 tok/s(대역폭 한계의 약 12%)라서
지금 코드를 어떤 클라우드 GPU로 옮겨도 약 2배뿐이다.

| 구현 상태 | A40 1,500문항 예상(판단, ±50%) |
|---|---|
| 현재 코드(순차 HF) | 165–260 h, 97–153달러 |
| HF 배치 생성(문항 4–8개 × n표본 한 번에) | 33–44 h, 19–26달러 |
| vLLM 샘플러 + 병합 가중치 push | 20–23 h, 12–13달러 |

GPU 선택(2026-10-30 RunPod 실시간 조회): **A40 48GB Secure 0.59달러/h, 재고 HIGH**가 가성비 최선.
RTX A6000 같은 가격이나 재고 LOW, 4090 Secure 0.89, A100 SXM 1.79, H100 SXM 3.99. Community A40은 재고 없음.
pod 생성 시 CUDA는 정확 일치가 아니라 `minCudaVersion 12.8`(대부분 호스트가 13.x).

구현 변경 목록(GPU 활용도 영향 순, 모두 로컬 구현·CPU/NF4 검증 가능, BF16 GPU 성공은 RunPod 단계에서):
1. vLLM 경로를 repair 파이프라인에 연결: `on_completion` 지원(작음), V59 protocol·24576 문맥·메모리 설정 연결(중간),
   문항 간 wave 스케줄러(T1 전체 → 채점 → T2 전체 …, 큼). 기존 `VLLMRolloutEngine`은 Vision LoRA·Merger를
   BF16으로 병합해 push하므로 vLLM 0.11의 Vision LoRA 미지원을 우회한다. 단 GPU에서 끝까지 돈 적 없음.
2. `local_train`·reload·eval의 BF16 정책 경로(`policy.build_policy` 재사용, NF4 전용 검사 일반화).
3. vLLM 보류 시 대안: n표본을 한 번의 HF generate로(행별 RNG 또는 새 seed 계약 — 샘플별 정확 재현 상실, 사용자 결정).
4. update당 문항 수 설정(현재 pair 2·10문항·5update 하드코딩), k update마다 저장·probe.
5. 증거 VJP는 감사 step에서만, 48GB에서는 Adam parking 생략, 베이스 해시를 pair 루프 밖으로.
6. 대규모용 비결정 커널 profile(게이트에서는 결정론 유지, 사용자 승인 필요).
7. 슬롯 없는 데이터용 **정답 검증 경로**(새 protocol): host가 `required_slots==()`일 때만 지정, Vision 동결
   Merger+LLM 학습, `VISION_OFF`가 아닌 `ANSWER_VERIFIED`로 기록, 답 보상 + 재시도, 노트는 생성·기록하고
   텍스트로 확인 가능한 항목만 보상.
8. 20시간 실행 대비 이어하기(체크포인트에 이미 optimizer/RNG/cursor 있음) 또는 300문항 단위 분할.

Pod 대기시간 줄이기(측정 12–46분 → 목표 6–12분, 판단): 전체 CPU 검사·데이터 오염검사·manifest는 로컬에서 끝내고,
clean clone으로 만든 exact-lock venv tarball과 고정 모델 13파일을 Secure network volume(약 40GB, 월 약 2.8달러)에
두며, Pod에서는 해시 확인 + 5분 이내 GPU smoke(BF16 연산, sync probe, n4 생성, 작은 update, 새 프로세스 reload) 후
바로 시작한다. 로컬 venv·모델 업로드는 §2.6에 따라 금지이므로 volume은 Pod 위에서 clean clone으로 만든다.
`HF_XET_HIGH_PERFORMANCE=1`, 확인 후 `HF_HUB_OFFLINE=1`.

예상 총비용(A40 Secure, vLLM 경로 수정 후): 통과율 필터 n8 약 1.8달러 + 파일럿 300 약 2.4–2.8 + 본 1,500 약 12–13
+ 시작·smoke 약 1 = **약 17–19달러, 약 30 GPU시간(판단, ±50%)**. vLLM 실패 시 HF 배치로 약 30–45달러.
본 실행 전 30분 A40 보정 측정(batch 16/32/44 tok/s, update 시간, 결정론 모드 attention 커널, HF+vLLM 동시 peak)으로 재산정한다.
사용자 결정이 필요한 protocol 변경: 배치 샘플링(seed 재현 상실), vLLM 샘플러(수치 차이, MIS로 보정), BF16 시작 정책,
one-step off-policy 겹치기. 각각 새 최소시험을 다시 한다.

## 8. 남은 위험과 미완료 항목

<!-- REMAINING -->

## 9. 출처

- VL-Rethinker https://arxiv.org/html/2504.08837v3 · ThinkLite-VL https://arxiv.org/html/2504.07934 ·
  SophiaVL-R1 https://arxiv.org/html/2505.17018 · Continual RFT https://arxiv.org/html/2507.05386v2 ·
  MM-Eureka https://arxiv.org/html/2503.07365 · MiMo-VL https://arxiv.org/html/2506.03569 ·
  RL's Razor https://arxiv.org/abs/2509.04259 · DAPO https://arxiv.org/html/2503.14476 ·
  Qwen3-VL report https://arxiv.org/html/2511.21631v2 · Qwen MMMU eval https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu ·
  LoRA Without Regret https://thinkingmachines.ai/blog/lora/ · EasyR1 https://github.com/hiyouga/EasyR1 ·
  TRL GRPO https://huggingface.co/docs/trl/main/en/grpo_trainer · Unsloth VLM RL https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl
- 데이터: ScienceQA https://github.com/lupantech/ScienceQA · AI2D https://arxiv.org/abs/1603.07396 ·
  TQA https://prior.allenai.org/projects/tqa · Geometry3K https://github.com/lupantech/InterGPS ·
  TabMWP https://github.com/lupantech/PromptPG · ChartQA https://arxiv.org/abs/2203.10244 · PlotQA https://arxiv.org/abs/1909.00997 ·
  MapQA https://arxiv.org/pdf/2211.08545 · VQA-RAD https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6244189/ ·
  SLAKE https://ar5iv.labs.arxiv.org/html/2102.09542 · PathVQA https://arxiv.org/abs/2003.10286 ·
  SSMR https://arxiv.org/html/2509.04059v2 · WikiArt https://huggingface.co/datasets/huggan/wikiart ·
  제외 근거: PMC-VQA https://arxiv.org/html/2305.10415 · ArxivQA https://arxiv.org/abs/2403.00231 ·
  MathV360K https://arxiv.org/html/2406.17294 · ViRL39K https://huggingface.co/datasets/TIGER-Lab/ViRL39K
- 신뢰도: 데이터 라이선스 일부(AI2D, FigureQA, DVQA, MapQA, SSMR) 미검증. MMMU 후보 겹침은 아직 실측 전.
