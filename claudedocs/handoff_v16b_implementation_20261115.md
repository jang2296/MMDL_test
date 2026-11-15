# v16b 구현·검증 인계

작성일: 2026-11-15. 정본: `plan_v16b_self_note_rl_20261115.md`.

현재 범위는 코드 구현, CPU 검사, 짧은 로컬 NF4 기능 확인이다. 게시와 RunPod 실행은 Claude Code가 맡는다. 이 문서는 실제 GPU 학습 효과나 MMMU 성적을 인증하지 않는다.

## 1. 자료

- 원 선정 산출물: `/home/jang2/mmdl-artifacts/self-note-single-20261105/v16b/data-curation-20261115/`.
- 조립된 입력: `/home/jang2/mmdl-artifacts/self-note-single-20261105/v16b/implementation-20261115/data/assembled/`.
- 본학습 3,000 / 감시 200 / 준비 SFT 후보 512. 가중치 학습에 사용할 수 있는 고유 문제는 최대 3,512개이고, 감시까지 포함한 자료 총수는 3,712개다. 준비 512개가 모두 SFT 행으로 채택되는 것은 아니다. 학습·감시·순서 파일의 원 SHA256은 유지했다.
- 준비 집합은 학습·감시·기존 보류분과 계열·그림이 겹치지 않는다. 남은 후보가 있는 7개 출처 안에서 학습 비중을 정규화하고 해시 순서로 선별했다.
- 준비 출처: MapQA 232, ChartQA 116, ScienceQA 92, DeepWeeds 36, NCTCRC 13, PlantVillage 13, TQA 10.
- 준비 사람 풀이 참조 92개. 실제 SFT 행 수는 원본 생성과 길이·정답·출처 검사를 마친 뒤 정한다.
- 준비 집합의 지원되는 OBS 문항은 0개다. 원본 관측 점수 0.5 같은 상수를 만들어 넣지 않는다. 처음 만나는 층의 OBS 이득은 시작값을 확보할 때까지 0이다.
- 학습 집합의 원 주석 참조 668개와 기존 지원 표기 192개는 다르다. 192개 중 chart 1개를 제외한 실제 OBS 적격은 **191개**(도형 47·그래프 96·장면 48)이며, 191개 모두 관측 프롬프트 생성을 CPU에서 확인했다. OBS 부적격 문항도 본학습 첫 풀이 3,000개에서 제외하지 않는다.
- 학습 목록 3,000개는 전수 프로그램 검사와 400개 표본 AI 시각 검토를 거쳤다. 전수 사람 검수가 아니다.

## 2. 설계 충돌을 해소한 구현 기준

1. §3.1 표의 준비 집합이 학습 3,000 안이라는 문장은 오래된 서술이다. §0·§8·§17에 따라 별도 512개를 사용한다.
2. 기준값은 생성 요청을 넣을 때 봉인한다. 지연 1 일정에서 묶음 c의 첫 풀이를 넣을 때 최신 확정 이력은 c−2까지다. 생성 이후 c−1 결과를 보고 기준값을 다시 쓰지 않는다. cutoff와 이력 해시를 보존한다.
3. 체크포인트에서는 다음 요청을 넣기 전에 현재 요청을 모두 끝낸다. 완료됐지만 아직 학습하지 않은 출력은 원문과 해시로 저장한다. 그 보상을 예정된 묶음보다 먼저 기준값에 반영하지 않는다.
4. LoRA 판의 과거 LLM 값과 현재 공유 Vision/Merger를 따로 확인한다. 옛 LLM 판에 현재 시각 경로를 연결하는 설계이며, 옛 체크포인트 전체와 현재 시각 값을 같다고 인증하지 않는다.
5. 표본 확률 μ는 생성한 토큰의 전체 어휘 처리 확률이다. TAIL 진단에 필요한 후보 확률은 로깅만 한다. 128개 후보를 받아 top-k 경계 동점까지 포함하고, 완전성을 증명하지 못하거나 nucleus 경계 동점 때문에 판정이 모호하면 명시적으로 실패한다. 후보를 답 선택이나 RL 재정규화에 쓰지 않는다.
6. 두 판 공존 검사는 첫 두 갱신에 추가 drain을 둔다. 진짜 학습 출력은 버리지 않고 버퍼로 돌려주며, 고정 합성 탐침의 단독·혼합 요청을 비교한다. 탐침은 학습 행·정답률에 들어가지 않는다.
7. 학습 도중 KV 크기·정밀도·이미지·토큰을 바꾸지 않는다. BF16 시작 전에 합성 FULL 역전파로 메모리를 측정하고, 고정 입력 상한 안에서만 실행한다. 긴 NOTE 입력을 고려해 설계의 약 4,000토큰 가정을 그대로 최대치로 인증하지 않는다.
8. 준비와 RL은 optimizer와 전역 LR 일정을 공유한다. RL 판 번호는 준비 갱신 수와 별개다. HOLD와 무신호 flush도 일정 슬롯은 소비하지만 AdamW는 실행하지 않는다.
9. 새 로컬 시험에서 노트나 C 행이 통과하지 않으면 그대로 0을 보고한다. CPU fixture의 노트 성공을 실제 Qwen의 노트 학습으로 바꾸어 말하지 않는다.
10. 감시나 자금 보호로 정지하면 평가를 자동 실행하지 않는다. 최종 정상 완료만 기존 단일 900 평가 준비 경로로 연결한다.

## 3. 요구사항과 구현 위치

| 설계 항목 | 구현 |
| --- | --- |
| 새 계약·기존 해시 보존·단계별 seed | `protocol.py`, `contract.py`, `single.py` |
| 학습·감시·준비 자료 분리와 단일 vault | `data_v16b.py` |
| 전체 어휘 처리 확률·벡터화 presence·NTF | `rl.py`, `train_v16b.py` |
| 봉인된 보상·기준값·이득 재검사 | `v16b_rows.py`, `v16b_runtime.py`, `train_v16b.py` |
| native 관측 F1 | `v16b_rows.py`, 기존 `content_v16.py` 재사용 |
| 정확한 주장·문자/UTF-8 구간·누설 차단 | `v16b_rows.py`, 기존 `self_note_v16.py` 재사용 |
| 틀린 내용·인용·철회·교훈 제외 마스크 | `v16b_rows.py` |
| RL/OBS/N/C/PREP 인코딩과 실제 손실 | `train_v4.py`, `train_v16b.py` |
| F0 첫 4토큰 앵커·경로별 실제 기울기 | `train_v16b.py` |
| 준비 512·원본 감시·행 봉인 | `prep_v16b.py` |
| 준비→RL·단일 optimizer·후속 감시 | `campaign_v16b.py`, `single_campaign.py` |
| 지연 1 요청 일정·마지막 세 flush | `v16b_runtime.py` |
| 요청별 LoRA·최대 두 판·시각 push | `single_vllm.py`, `llm_lora.py` |
| 로컬 순차 생성·과거 판 보존 | `collect.py`, `v16b_runtime.py` |
| drain 저장·미소비 결과·되돌림 | `v16b_runtime.py`, `campaign_v16b.py`, 기존 checkpoint |
| 갱신별 손실·기울기·메모리·시간·재풀이 탈락 사유 보존 | `single.py`, `campaign_v16b.py`, `v16b_runtime.py`, `v16b_rows.py` |
| 길이·정답률·정오 누락·TAIL·IS 감시 | `guard_v16b.py` |
| 자금 보호·충전 대기·평가 6.5시간 | 비공개 handoff의 guard/funding 스크립트 |
| 비병합 내보내기·v11 단일 900 사전등록 | `proof_eval.py`, `campaign_v16b.py` |
| 전체 작업 연결·보완 판정 별도 | `plan.py`, 비공개 `cv_stage_v16b.py` |
| A100 SXM만 허용·실제 MCP 실행 분리 | 비공개 `create_pod_v16b.py` |

경로는 별도 표시가 없으면 `fine_tuning/error_note/` 아래다. `proof_eval.py`는 `fine_tuning/notebook_rl/`에 있다.

## 4. CPU 검증

집중 검사는 각 구현의 실제 unittest로 수행했다. 종합 검사와 로컬 실제 실행 결과는 아래 검증 원장에 기록한다.

- 기존 v11·v12·v13·v14·v15·content-v16·v2의 golden protocol 해시를 다시 생성해 일치 확인했다.
- 작은 무작위 Qwen CPU 모델로 실제 손실·기울기·HOLD·마이크로배치·비병합 내보내기를 검사했다. 실제 사전학습 Qwen 실행은 아니다.
- 독립 검토에서 장면 원본 순서와 공간 순서 불일치, 부정된 등식에 대한 거짓 교정 허용을 발견했다. 공간 순서를 명시한 1부터의 번호로 통일하고 모호한 번호·동일 x 좌표·부정·철회 주장을 미검증 처리하도록 수정했다.
- 새 주장은 근거가 있을 때만 VERIFIED다. 자연어 문법의 지원 범위 밖은 UNVERIFIABLE로 남는다.

## 5. 실행 전 남는 증거

현재 실제 동결 F0 기록 재표집 2,000회에서 측정 가능한 규칙의 합산 거짓 정지는 1.5%다. 기준 2% 이내지만 **전체 인증은 INCOMPLETE**다. 저장된 F0 기록에는 TAIL_RATE와 IS_DRIFT를 계산할 확률·로그가 없다. null을 0으로 바꾸지 않았고 문턱을 맞추어 낮추지 않았다.

이 상태를 전체 감시 인증 PASS로 간주해 cloud 사전 관문을 통과시키면 안 된다. Claude Code는 해당 규칙의 실제 증거를 확보하거나 현재 제한을 사용자에게 설명해 사전 관문의 범위를 명시적으로 정해야 한다. GPU에서만 확인 가능한 PF·메모리·두 판 공존 검사는 실행 코드가 있더라도 실제 RunPod 결과 전에는 미검증이다.

비공개 실행 도움 파일은 `/home/jang2/mmdl-artifacts/self-note-single-20261105/v16b/implementation-20261115/handoff/`에 있다. `eval-guard-plan.md`가 단계와 외부 잔액 공급 계약을 설명한다. 생성 도우미는 MCP 요청 본문만 준비하며 스스로 포드를 만들거나 삭제하지 않는다.

## 6. 검증 원장

구현·CPU 통합 검사와 짧은 실제 NF4 갱신·저장·재로딩 확인을 마쳤다. 사용자가 본학습 3,000개 + 별도 준비 SFT 후보 512개 구성을 재확인했다. 실제 노트·재풀이의 양성 학습과 BF16·vLLM·정확도 개선은 아직 인증하지 않았다.

- 원장 루트: `/home/jang2/mmdl-artifacts/self-note-single-20261105/v16b/implementation-20261115/`.
- 실제 3,712개 입력 계약 사전 검사와 191개 OBS 프롬프트 생성 통과. 증거: `campaign-cpu-check-v2/prep/data-contract-preflight.json`.
- CPU 회귀 첫 실행은 `PYTHONPATH=src` 누락에 따른 import 오류가 있었고, 재실행에서 새 버전의 정확한 기대값을 반영하지 않은 테스트 4건이 실패했다. 실패 로그는 보존하고 import 경로 및 기대값을 수정했다. 마지막 관련 4개 모듈은 **50개 검사 OK** (`checks/cpu-regression-3.log`).
- 세 실행의 중복과 실패한 import placeholder를 제외한 고유 테스트 1,264개 중 **1,259개 통과, 5개 skip**이다. 처음 전체 검사 후 집중 재검사로 실패를 해소한 결과이며, 단일 전체 실행 PASS는 아니다. skip은 vLLM GPU 검사 3개, 실제 v14 체크포인트 검사 1개, 해당 로컬 사전등록 파일 부재 검사 1개다.
- 이후 로컬 이미지 계약 검사 1개·재풀이 거부 사유 2개·시간 기록/되돌림 2개를 추가했다. 최종 합계는 **고유 1,269개 중 1,264개 통과·5개 skip**이다. 최종 변경 통합 78개 검사도 통과했다. 증거: `checks/cpu-consolidated.json`, `checks/cpu-final-metrics.log`.
- 마지막 검사 명령: `PYTHONPATH=/mnt/c/projects/multimodel/src CUDA_VISIBLE_DEVICES='' /home/jang2/mmdl-envs/notebook-nf4-portability-v1/bin/python -m unittest fine_tuning.tests.test_error_note_single_eval fine_tuning.tests.test_error_note_v16b_eval fine_tuning.tests.test_error_note_v16b_prep fine_tuning.tests.test_error_note_v16b_campaign -q`.
- 변경 범위 Ruff 통과. 신규·연결 모듈 10개 범위 mypy 통과. 마지막 변경 4개 모듈도 `--follow-imports=skip --ignore-missing-imports`로 통과했다(`checks/mypy-final-metrics.log`). 기존 `protocol.py`·`single.py`를 포함한 더 넓은 검사에는 기존 부분의 타입 오류 19건이 남아 있으며 전체 타입 검사 통과로 보고하지 않는다.
- 최종 로컬 실행용 동결 소스 256개 Python 파일은 상위 실제 파일과 해시 일치. `source-freeze-5.json` 파일 SHA256: `c1ad651400c5d257a55e8ba07d3e10e9fc4072b8f21572ceadbd4d9a53913c81`. 실제 실행 소스는 `source-root-v5/`다.
- 교수님 제출용 `MMDL/` clean 확인, `AGENTS.md`·`CLAUDE.md` byte 동일. 게시·클라우드·모델 다운로드 없음.

## 7. 실제 로컬 시험과 현재 한계

최종 영수증: `/home/jang2/mmdl-artifacts/self-note-single-20261105/v16b/implementation-20261115/local-smoke-v4/functional-receipt.json`.

- 기존 Qwen3-VL-4B NF4 캐시만 사용하고 Hugging Face와 Transformers를 offline으로 고정했다. 7B 다운로드는 없다.
- 준비 SFT 1갱신 + 2문항씩 4묶음 + 지연 처리 3슬롯을 끝냈다. 총 일정 슬롯 8개, 실제 optimizer 갱신 3회(준비 1·RL 2). 무신호 슬롯에는 optimizer를 실행하지 않았다.
- 실제 생성: 첫 풀이 8개, OBS 3개, 노트 2개. 준비 SFT는 데이터셋 사람 풀이 2행이며 자기생성 준비 풀이의 수용은 0이었다.
- 노트 2개는 정상 종료했지만 `lesson`을 객체 대신 문자열로 써 `NOTE_SCHEMA`에서 탈락했다. **N 행 0, 교훈 재풀이 0, C 행 0**이다. 마지막 flush의 C 행도 0으로 기록했다. 노트나 응답을 고쳐서 통과시키지 않았고 검사 조건을 완화하지 않았다.
- 따라서 실제 모델에서 노트 교정 SFT와 재풀이 학습이 작동했다는 양성 증거는 없다. 이 경로의 검증·손실·기울기·일정 연결은 CPU 시험에서 확인했으며, 실제 모델의 성공과 구분한다.
- 준비 체크포인트 → 최종 체크포인트에서 언어 504·시각 96·Merger 24개, 총 624개 학습 텐서의 해시가 변했다. 그림 읽기 정확도 향상 증거는 아니다.
- 학습 PID 450853 → 별도 재로딩 PID 451392. optimizer·scheduler·RNG 일치, 탐침 로그확률 최대 차이 0, greedy token 일치.
- 갱신 영수증 8개(`campaign/update-results/slot-*.json`)와 RL/flush 시간 영수증 7개(`campaign/runtime/update-timing-*.json`)의 해시를 검사했다. 실제 CUDA 최대 allocated 기록은 6,174,287,872 bytes다. 물리 VRAM 상주·A100 용량·BF16 인증으로 확대하지 않는다.
- 시간 기록은 host wall time이다. `sampler.step`에서 기다린 시간은 생성 시간에 포함하며, 별도 idle wait가 없는 현재 일정에서 idle은 0이다. GPU 사용률이나 순수 장치 실행 시간 측정이 아니다.
- 갱신 영수증은 손실·채널별 토큰·이득·NTF/IS·HOLD·N 기울기와 실제 update 기울기·메모리를 보존한다. 재풀이 결과는 수용/탈락 이유를 원문 해시에 묶는다. 저장 지점 이후 결과는 재개 시 덮어쓰지 않고 rollback 보관함으로 옮긴다.
- 첫 확인은 669.6초, 최종 소스 확인은 551.5초였다. 중간 점검을 포함한 두 확인의 경과 시간은 **1,452.7초(약 24.2분)**로 30분 이내였다. 최종 확인은 같은 F0 준비 생성 자료를 재사용했고 학습 가중치는 F0에서 새로 시작했다.
- 첫 모델 로드 전 시도는 옛 v16 이미지 사전에 있던 추가 키와 v16b 계약의 불일치로 차단됐다. 공유 v16b 이미지 상수로 연결을 수정했고 실패 원장 `local-smoke-v2/`를 보존했다. 이후 첫 성공본 `local-smoke-v3/`에서 통계 보존 누락을 확인해 최종 소스로 다시 확인했다. 좋은 점수/답을 골라 채택한 실행이 아니다.
- `require_complete()`도 최종 checkpoint·monitor·cursor를 확인했다. 이는 로컬 기능 완료이며 실제 MMMU 평가 실행 승인이 아니다.

현재 Codex 작업은 여기서 대기한다. Claude Code는 §5의 감시 인증 공백과 BF16 메모리·vLLM 두 판 공존 관문을 해결한 뒤 게시·RunPod 실행을 맡는다. 로컬 초소형 vLLM GPU 시험은 총 시간 제한 안에서 실행하지 않았으며 미실행으로 남긴다. V8 65점 달성 여부는 아직 측정하지 않았다.
