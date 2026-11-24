# DevMate 🤖

**공식 문서 기반으로 개발자 질문에 한국어로 답하는 RAG 챗봇**

![Python](https://img.shields.io/badge/Python-3.10%20~%203.12-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Status](https://img.shields.io/badge/Status-개발%20중-yellow.svg)

공식 문서는 대부분 영어이고 양이 많아서, 필요한 내용을 찾아 이해하는 데 시간이 걸립니다.
DevMate는 문서를 미리 쪼개서 벡터 DB(FAISS)에 넣어 두고, 질문이 들어오면 관련 부분을 찾아 LLM이 한국어로 설명하게 합니다. 답변마다 근거가 된 문서를 함께 보여 줘서 지어낸 답인지 확인할 수 있습니다.

**LLM과 임베딩을 모두 로컬 [Ollama](https://ollama.com)로 돌리기 때문에 API 키나 사용료 없이 동작하고, 질문·문서가 외부로 나가지 않습니다.**

---

## 주요 기능

- **문서 검색 + 답변 생성(RAG)**: 질문과 비슷한 문서 조각 top-3을 찾아 LLM에 넘기고, 출처와 함께 답변
- **LLM 선택**: 이 PC의 Ollama에 설치된 모델 중 사이드바에서 골라 사용 (llama3.2, qwen, gemma 등)
- **대화 맥락 유지**: 같은 세션 안에서는 이전 질문을 이어서 답변 (`query_with_conversation`)
- **Streamlit 웹 UI**
  - 기술별 문서 목록 탐색 (Backend / Frontend / DB / DevOps)
  - 카테고리별 예시 질문 클릭으로 바로 질문
  - 대화 초기화
- **문서 처리**: PDF·DOCX·TXT 로드 → 청크 분할 → 로컬 임베딩(bge-m3) → FAISS 저장
- **공식 문서 다운로드 스크립트**: FastAPI·LangChain·Python 문서 자동 수집

### 수록 문서

`data/source_docs/`에 22개 기술, 140개 문서가 있고 모두 검색 인덱스에 들어 있습니다.

| 분류 | 기술 |
|---|---|
| Backend | Python, FastAPI, Django, Spring Boot, Node.js, NestJS, LangChain |
| Frontend | React, JavaScript, TypeScript, HTML/CSS, Tailwind CSS |
| Database & Cache | MySQL/SQL, MongoDB, Redis |
| DevOps & Infra | Docker, AWS, AWS 인프라(VPC·ALB·ECS·Route 53·CloudFront), Nginx, Git, Flutter |

---

## 기술 스택

| 영역 | 사용 기술 |
|---|---|
| LLM | 로컬 Ollama (기본값 `llama3.2`, UI에서 설치된 모델 선택) |
| 임베딩 | 로컬 Ollama `bge-m3` (한국어·영어 다국어 임베딩) |
| RAG 프레임워크 | LangChain 0.1 |
| 벡터 DB | FAISS (CPU) |
| UI | Streamlit |
| 문서 파싱 | pypdf, pdfplumber, python-docx |
| 설정 | pydantic-settings, python-dotenv |
| 테스트 | pytest, pytest-asyncio |

---

## 구조

레이어드 아키텍처로 나눠 두어서 LLM이나 벡터 DB를 바꿀 때 해당 계층만 고치면 됩니다.

```
devmate/
├── app.py                        # Streamlit 웹 UI (실행 진입점)
├── demo.py                       # 터미널용 데모
├── app/
│   ├── config.py                 # .env 설정 로드
│   ├── domain/                   # 모델·스키마·예외 (Document, Chunk, Conversation 등)
│   ├── application/
│   │   ├── document_processor.py # 문서 → 청크 → 임베딩 → 저장
│   │   └── query_handler.py      # 검색 + 프롬프트 + LLM 호출
│   └── infrastructure/
│       ├── llm_client.py         # OllamaClient, PromptBuilder
│       ├── embeddings.py         # 로컬 임베딩(bge-m3) 생성
│       ├── vector_store.py       # FAISSVectorStore
│       └── text_splitter.py      # 청크 분할, PDF/DOCX/TXT 로더
├── scripts/
│   ├── download_docs.py          # 공식 문서 다운로드
│   └── init_index.py             # 문서 인덱싱 → FAISS 저장
├── data/
│   ├── source_docs/              # 원본 문서 (기술별 폴더)
│   └── vector_store/             # FAISS 인덱스
└── tests/                        # unit / integration / e2e
```

---

## 시작하기

### 1. 준비물

- Python 3.10 ~ 3.12
- [Ollama](https://ollama.com/download)
- API 키는 필요 없습니다.

### 2. 설치

```bash
git clone https://github.com/jhpark-coder/devmate.git
cd devmate

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Ollama 모델 두 개를 받아 둡니다.

```bash
ollama pull bge-m3      # 검색용 임베딩 (약 1.2GB)
ollama pull llama3.2    # 답변용 (이미 다른 모델이 있으면 생략 가능)
```

모델 이름을 바꾸고 싶으면 `.env.example`을 `.env`로 복사해서 `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`을 고치면 됩니다.

### 3. 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`이 열립니다. 사이드바에서 LLM을 고르고 질문을 입력하거나 예시 질문을 누르면 됩니다.

터미널에서 바로 써 보려면:

```bash
python demo.py
```

### 4. 인덱스 다시 만들기 (선택)

문서를 추가하거나 임베딩 모델을 바꿨을 때만 필요합니다. `data/source_docs/` 아래 폴더를 전부 다시 인덱싱합니다.

```bash
python scripts/init_index.py
```

자세한 사용법은 [USAGE.md](USAGE.md)에 있습니다.

---

## 테스트

```bash
pytest                       # 전체
pytest tests/unit -v         # 단위 테스트만 (Ollama 불필요)
pytest --cov=app             # 커버리지
```

integration 테스트는 Ollama와 `bge-m3`가, e2e 테스트는 추가로 `OLLAMA_MODEL` 모델이 있어야 돌고, 없으면 자동으로 건너뜁니다.

---

## 진행 상황

- [x] 도메인 모델, FAISS 벡터 스토어
- [x] 문서 처리 파이프라인, 질의응답
- [x] 공식 문서 자동 다운로드·인덱싱
- [x] Streamlit 웹 UI, Ollama 연동
- [x] 문서 140개, 22개 기술로 확장
- [x] OpenAI 제거 — LLM·임베딩 모두 로컬 Ollama로 전환, 140개 전체 인덱싱

### 할 일

- [ ] 대화 기록 저장 (SQLite)
- [ ] 한영 병행 답변 모드
- [ ] FastAPI REST API (`app/api/` 자리만 있음)
- [ ] Streamlit Cloud 배포

---

## 만든 사람

**jhpark-coder** · [GitHub](https://github.com/jhpark-coder)
