# DevMate 사용 가이드

## 🚀 빠른 시작

DevMate는 FastAPI, LangChain, Python 공식 문서를 기반으로 개발자 질문에 답변하는 AI 어시스턴트입니다.

### 1단계: 환경 설정 (최초 1회)

```bash
# 가상환경 활성화
./venv/Scripts/activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 환경 변수 확인
# .env 파일에 OPENAI_API_KEY가 설정되어 있어야 함
```

### 2단계: 공식 문서 다운로드 (최초 1회)

```bash
python scripts/download_docs.py
```

**다운로드되는 문서:**
- ✅ FastAPI: 6개 핵심 문서 (Tutorial, Path/Query Parameters, Request Body, Dependency Injection, Async)
- ✅ LangChain: 4개 핵심 문서 (Introduction, Chains, Vector Stores, Retrieval QA)
- ✅ Python: 3개 핵심 문서 (asyncio, Type Hints, Dataclasses)

**저장 위치:** `data/source_docs/`

### 3단계: 문서 인덱싱 (최초 1회)

```bash
./venv/Scripts/python scripts/init_index.py
```

**처리 결과:**
- 총 13개 문서 → 25개 청크로 분할
- OpenAI 임베딩 생성
- FAISS 벡터 DB에 저장
- 저장 위치: `data/vector_store/production_index/`

### 4단계: DevMate 사용

```bash
./venv/Scripts/python demo.py
```

---

## 💬 사용 예시

### 시나리오 1: FastAPI 질문

```
질문: FastAPI에서 async 함수는 어떻게 정의하나요?

답변:
FastAPI에서 async 함수는 async def를 사용하여 정의합니다:

```python
@app.get("/")
async def read_root():
    return {"Hello": "World"}
```

출처: FASTAPI - Async Await
```

### 시나리오 2: LangChain 질문

```
질문: LangChain에서 vector store는 어떻게 사용하나요?

답변:
LangChain에서 FAISS vector store를 사용하는 방법:

```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

# 검색
docs = docsearch.similarity_search(query)
```

출처: LANGCHAIN - Vector Stores
```

### 시나리오 3: Python 질문

```
질문: Python의 dataclass는 어떻게 쓰나요?

답변:
Dataclass는 @dataclass 데코레이터를 사용합니다:

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(1.5, 2.5)
```

출처: PYTHON - Dataclasses
```

### 시나리오 4: 대화 컨텍스트 유지

```
질문 1: FastAPI에서 dependency injection은 뭔가요?
답변: [Depends() 사용 방법 설명]

질문 2: 그럼 class를 dependency로 쓸 수 있나요?
답변: [이전 대화 기억하고 class dependency 설명]

질문 3: 예제 코드 보여줘
답변: [구체적인 예제 코드 제공]
```

---

## 🎯 주요 명령어

### demo.py 명령어

- **질문 입력**: 그냥 질문 입력하고 Enter
- **새 대화**: `new` 입력
- **종료**: `quit` 또는 `exit` 입력

### 추가 문서 업로드

demo.py 실행 중:
```
추가 문서를 업로드하시겠습니까? (y/n): y
파일 경로를 입력하세요: C:\Users\...\my_doc.pdf
```

---

## 📁 프로젝트 구조

```
chatbotProject/
├── scripts/
│   ├── download_docs.py      # 공식 문서 다운로더
│   └── init_index.py          # 초기 인덱싱
├── data/
│   ├── source_docs/           # 다운로드된 공식 문서
│   │   ├── fastapi/          # FastAPI 문서 (6개)
│   │   ├── langchain/        # LangChain 문서 (4개)
│   │   └── python/           # Python 문서 (3개)
│   └── vector_store/
│       └── production_index/  # 벡터 DB 인덱스
├── demo.py                    # 데모 스크립트
└── .env                       # OpenAI API 키
```

---

## 🔧 고급 사용법

### 특정 소스만 다운로드

```bash
# FastAPI만
python scripts/download_docs.py --source fastapi

# LangChain만
python scripts/download_docs.py --source langchain

# Python만
python scripts/download_docs.py --source python
```

### 문서 업데이트

문서를 다시 다운로드하고 재인덱싱:

```bash
# 1. 기존 인덱스 삭제
rm -rf data/vector_store/production_index

# 2. 문서 재다운로드
python scripts/download_docs.py

# 3. 재인덱싱
./venv/Scripts/python scripts/init_index.py
```

### 인덱스 초기화

인덱스만 다시 생성:

```bash
rm -rf data/vector_store/production_index
./venv/Scripts/python scripts/init_index.py
```

---

## ❓ 문제 해결

### "인덱스가 없습니다" 에러

```bash
# 해결 방법
python scripts/download_docs.py
./venv/Scripts/python scripts/init_index.py
```

### "OPENAI_API_KEY not set" 에러

```bash
# .env 파일 확인
cat .env

# OPENAI_API_KEY가 설정되어 있는지 확인
# 없으면 추가:
# OPENAI_API_KEY=sk-...
```

### "No relevant documents found" 에러

인덱스가 비어있거나 질문과 관련된 문서가 없습니다.

```bash
# 인덱스 재생성
./venv/Scripts/python scripts/init_index.py
```

---

## 📊 현재 지원 문서

| 카테고리 | 문서 수 | 주요 내용 |
|---------|---------|----------|
| FastAPI | 6개 | Tutorial, Parameters, Request Body, DI, Async |
| LangChain | 4개 | Introduction, Chains, Vector Stores, QA |
| Python | 3개 | asyncio, Type Hints, Dataclasses |
| **총계** | **13개** | **25개 청크** |

---

## 🎉 다음 단계

현재는 **로컬 CLI 데모** 단계입니다.

**Phase 2에서 추가 예정:**
- 📡 FastAPI REST API 서버
- 🖥️ Streamlit 웹 UI
- 💾 대화 히스토리 DB 저장
- 🔄 스트리밍 응답
- 🌐 더 많은 공식 문서 지원

---

## 📝 라이선스

MIT License
