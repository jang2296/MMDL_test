# 오답노트형 자기개선 파인튜닝 (`fine_tuning/error_note/`)

계획 정본: `claudedocs/research_self_note_finetuning_plan_20261101.md`. 규칙: `CLAUDE.md` §2.1, §3.0.

문항을 한 번 풀고(정답 없이), V8로 채점하고, 모델이 정답을 보고 오답노트와 고쳐 쓴 풀이를 쓴 뒤, 풀이와 노트를 다음 토큰 지도학습한다.
노트가 이미지 오독이라 하면 Vision 조정층 + Merger + LLM 조정층, 그 외에는 Merger + LLM 조정층을 갱신한다. 원본 가중치는 고정이다.
상위 모델 풀이, 튜터 모델, MMMU·MMMU-Pro 학습, 여러 답 생성 후 선택은 금지다.

## 파일

| 파일 | 내용 |
|---|---|
| `contract.py` | 고정값, 노트 프롬프트, 노트 파싱, 학습 행 생성과 거르기. 그래픽카드 불필요 |
| `order.py`, `item_order.json` | 결과를 보기 전에 고정한 3,417문항 순서. 앞 300개가 첫 시험 |
| `protocol.py` | 학습 설정 (조정층, 학습률, 정밀도) |
| `collect.py` | 풀이·노트 생성(`solve`, `note`, `sweep`), 학습 행 만들기(`rows`) |
| `train.py` | 학습(`train`), 새 프로세스 재로딩 검사(`reload`) |
| `plan.py` | 대여기 단계 계획 작성 |

재사용: `notebook_rl`의 `policy`, `checkpoint`, `export`, `input_preparation`, `prompts`, `answers`, `v8_parser`, `verifier_service`, `vault`, `dataset`, `guards`, `pod/`.
평가는 `notebook_rl/proof_eval.py`와 `pod/eval_stage.py`로 `src/mmdl` 평가 코드를 그대로 호출한다. 그래서 평가만은 저장소 전체가 필요하다.

## 고정값

P0 프롬프트, 이미지 262144–1310720 픽셀(`qwen_vl_utils 0.0.14`), 최대 8192 토큰, temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, seed `SHA256("3407:{문항ID}")`, 채점 `team-final-answer-v8`.

## 실행

```bash
# 로컬 4비트: 작동 확인 전용. 성능 근거로 쓰지 않는다
python -m fine_tuning.error_note.collect solve --build-root V --order fine_tuning/error_note/item_order.json --limit 2 --model-path SNAP --output OUT --engine hf
python -m fine_tuning.error_note.collect note  --build-root V --order fine_tuning/error_note/item_order.json --limit 2 --model-path SNAP --output OUT --engine hf
python -m fine_tuning.error_note.collect rows  --build-root V --order fine_tuning/error_note/item_order.json --limit 2 --tokenizer-path SNAP --output OUT
python -m fine_tuning.error_note.train train --rows OUT/rows.jsonl --build-root V --order fine_tuning/error_note/item_order.json --limit 2 --arm note --precision nf4 --model-path SNAP --output RUN --code-commit HEX
python -m fine_tuning.error_note.train reload --run RUN --model-path SNAP --precision nf4

# 런포드 16비트: 단계 계획을 만들어 pod/start.sh 작업으로 실행한다 (pod/RUNBOOK.md 맨 앞 절)
python -m fine_tuning.error_note.plan plan --stage note-300 --params PARAMS.json --output STEPS.json
```

## 격리

- 첫 풀이 프로세스는 정답 파일을 열 수 없다(프로세스 안 감사 훅, 운영체제 격리는 아님). 채점은 별도 프로세스의 채점 서비스가 한다.
- 노트 프로세스는 시작할 때 정답을 읽고 같은 훅을 건다. 학습 프로세스는 수집 파일만 읽는다.
- 모든 학습 단계에서 MMMU·MMMU-Pro 경로와 상위 모델 SDK를 막는다.

## 보고할 때

- 노트의 이미지 오독은 **자기 신고**다. 실제 시각 오류로 쓰지 않는다.
- 틀렸던 342문항은 학습 없이 다시 풀어도 일부 맞는다. 미학습 재실행과 같은 문항·seed로만 비교한다.
- 300문항 통과는 돈을 더 쓸지 정하는 문턱이지 개선 입증이 아니다.
