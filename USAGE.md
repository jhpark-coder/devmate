# DevMate 사용 가이드

설치는 [README](README.md#시작하기)를 먼저 따라 해 주세요. 이 문서는 설치 이후의 사용법과 문제 해결을 다룹니다.

---

## 웹 UI (권장)

```bash
streamlit run app.py
```

`http://localhost:8501`이 열립니다.

| 사이드바 | 하는 일 |
|---|---|
| 🧠 LLM 선택 | Ollama에 설치된 답변용 모델 중 고르기 (임베딩 전용 모델은 목록에서 빠짐) |
| 📚 지원 문서 | 기술을 누르면 그 기술의 문서 목록이 나오고, 문서를 누르면 관련 질문이 입력됨 |
| 💡 질문 예시 | 누르면 바로 질문 |
| 🗑️ 대화 초기화 | 대화 맥락을 지우고 새로 시작 |

같은 대화 안에서는 이전 질문을 기억하므로 "그럼 예제 코드도 보여줘"처럼 이어서 물어볼 수 있습니다.

---

## 터미널 데모

```bash
python demo.py
```

- 시작할 때 추가 문서(.txt, .pdf, .docx)를 올릴지 묻습니다.
- 질문을 입력하고 Enter
- `new`: 새 대화 시작
- `quit` / `exit` / `종료`: 끝내기

답변 모델은 `.env`의 `OLLAMA_MODEL`(기본값 `llama3.2`)을 씁니다.

---

## 설정 (`.env`)

`.env.example`을 `.env`로 복사해서 고칩니다. 없으면 아래 기본값으로 동작합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 서버 주소 |
| `OLLAMA_MODEL` | `llama3.2` | 답변 모델 (데모·테스트용, 웹 UI는 화면에서 선택) |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | 검색용 임베딩 모델 |
| `LLM_TEMPERATURE` | `0.0` | 0에 가까울수록 같은 질문에 같은 답 |
| `LLM_MAX_TOKENS` | `1500` | 답변 최대 길이 (토큰) |

> 임베딩 모델을 바꾸면 기존 인덱스와 벡터가 맞지 않으므로 반드시 `python scripts/init_index.py`로 인덱스를 다시 만드세요.

---

## 문서 추가하기

1. `data/source_docs/<기술이름>/` 폴더에 `.txt` 파일을 넣습니다. 새 기술이면 폴더를 새로 만들면 됩니다.
2. 인덱스를 다시 만듭니다.

```bash
python scripts/init_index.py
```

기존 인덱스는 지우고 `data/source_docs/` 아래 폴더를 전부 새로 인덱싱합니다.

FastAPI·LangChain·Python 공식 문서를 웹에서 다시 받고 싶으면:

```bash
python scripts/download_docs.py --all
python scripts/download_docs.py --source fastapi   # 하나만
```

---

## 문제 해결

### "❌ Ollama 연결 실패"

Ollama가 꺼져 있습니다. Ollama 앱을 실행하거나 `ollama serve`를 실행하세요.

### "임베딩 모델 'bge-m3' 을 찾을 수 없습니다"

```bash
ollama pull bge-m3
```

### "설치된 답변용 모델이 없습니다"

```bash
ollama pull llama3.2
```

### "인덱스가 없습니다"

```bash
python scripts/init_index.py
```

### 답변이 느릴 때

답변 속도는 고른 모델 크기와 PC 사양에 달려 있습니다. 사이드바에서 더 작은 모델(3B~8B)을 고르세요.

### 테스트가 전부 skipped로 나올 때

integration·e2e 테스트는 Ollama 모델이 없으면 건너뜁니다. 위의 `ollama pull`을 먼저 하세요.
