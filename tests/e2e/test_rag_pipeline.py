"""
E2E 테스트: 전체 RAG 파이프라인

문서 업로드부터 질의응답까지 전체 워크플로우를 테스트합니다.

테스트 시나리오:
1. 문서 업로드 및 인덱싱
2. 문서 검색
3. 질의응답 생성
4. 대화 컨텍스트 유지
"""

import pytest
import os
from pathlib import Path
import shutil

from langchain_openai import OpenAIEmbeddings

from app.application.document_processor import DocumentProcessor
from app.application.query_handler import QueryHandler
from app.infrastructure.vector_store import FAISSVectorStore
from app.infrastructure.llm_client import OpenAIClient
from app.infrastructure.text_splitter import CustomTextSplitter
from app.domain.models import Conversation, SourceType
from app.config import settings


# OpenAI API 키 확인
pytestmark = pytest.mark.skipif(
    not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "",
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def test_index_path(tmp_path):
    """임시 인덱스 경로"""
    return tmp_path / "test_faiss_index"


@pytest.fixture
def test_documents_path(tmp_path):
    """임시 문서 저장 경로"""
    docs_path = tmp_path / "documents"
    docs_path.mkdir(parents=True, exist_ok=True)
    return docs_path


@pytest.fixture
def embeddings():
    """OpenAI Embeddings 클라이언트"""
    return OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        model="text-embedding-ada-002"
    )


@pytest.fixture
def vector_store(embeddings, test_index_path):
    """테스트용 FAISS 벡터 스토어"""
    store = FAISSVectorStore(
        embeddings=embeddings,
        index_path=str(test_index_path)
    )

    yield store

    # Cleanup
    if test_index_path.exists():
        shutil.rmtree(test_index_path)


@pytest.fixture
def text_splitter():
    """테스트용 텍스트 분할기"""
    return CustomTextSplitter(
        chunk_size=500,  # 테스트용 중간 크기
        chunk_overlap=100
    )


@pytest.fixture
def document_processor(vector_store, text_splitter, test_documents_path):
    """문서 프로세서"""
    original_docs_path = settings.DOCUMENTS_PATH
    settings.DOCUMENTS_PATH = test_documents_path

    processor = DocumentProcessor(
        vector_store=vector_store,
        text_splitter=text_splitter
    )

    yield processor

    settings.DOCUMENTS_PATH = original_docs_path


@pytest.fixture
def llm_client():
    """LLM 클라이언트"""
    return OpenAIClient()


@pytest.fixture
def query_handler(vector_store, llm_client):
    """질의응답 핸들러"""
    return QueryHandler(
        vector_store=vector_store,
        llm_client=llm_client,
        similarity_top_k=3
    )


@pytest.fixture
def fastapi_docs_file(tmp_path):
    """FastAPI 문서 샘플"""
    file_path = tmp_path / "fastapi_guide.txt"
    content = """
# FastAPI 시작하기

FastAPI는 Python 3.7+ 기반의 현대적이고 빠른 웹 프레임워크입니다.

## 주요 특징

1. **빠른 성능**: NodeJS 및 Go와 동등한 매우 높은 성능
2. **빠른 개발**: 개발 속도를 약 200~300% 향상
3. **적은 버그**: 사람의 실수로 인한 버그를 약 40% 감소
4. **직관적**: 훌륭한 편집기 지원과 자동완성 기능
5. **표준 기반**: OpenAPI 및 JSON Schema 표준 완전 호환

## 설치 방법

FastAPI를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

## 첫 번째 API 만들기

`main.py` 파일을 생성하고 다음 코드를 작성하세요:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

## 서버 실행

다음 명령어로 서버를 시작할 수 있습니다:

```bash
uvicorn main:app --reload
```

`--reload` 옵션은 코드 변경 시 자동으로 서버를 재시작합니다 (개발 환경에서만 사용).

## 자동 대화형 API 문서

FastAPI는 자동으로 대화형 API 문서를 제공합니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

이 문서들은 자동으로 생성되며, API를 직접 테스트할 수 있습니다.

## 경로 매개변수

URL 경로에 매개변수를 포함할 수 있습니다:

```python
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}
```

FastAPI는 Python 타입 힌트를 사용하여 자동으로 매개변수를 검증합니다.

## 쿼리 매개변수

함수 매개변수로 쿼리 매개변수를 정의할 수 있습니다:

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

## 요청 본문

Pydantic 모델을 사용하여 요청 본문을 정의합니다:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}
```

## 비동기 처리

FastAPI는 `async/await`를 완벽하게 지원합니다:

```python
@app.get("/async-data")
async def get_async_data():
    data = await some_async_function()
    return {"data": data}
```

비동기 함수를 사용하면 I/O 대기 시간 동안 다른 요청을 처리할 수 있어 성능이 향상됩니다.
"""
    file_path.write_text(content, encoding='utf-8')
    return str(file_path)


class TestRAGPipeline:
    """전체 RAG 파이프라인 E2E 테스트"""

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self,
        document_processor,
        query_handler,
        fastapi_docs_file
    ):
        """
        완전한 워크플로우 테스트:
        1. 문서 업로드
        2. 질문 → 답변 생성
        3. 출처 확인
        """
        # Step 1: 문서 업로드 및 인덱싱
        print("\n" + "="*60)
        print("Step 1: 문서 업로드 및 인덱싱")
        print("="*60)

        document = await document_processor.upload_and_process(
            source_file_path=fastapi_docs_file,
            title="FastAPI 시작 가이드",
            source_type=SourceType.FASTAPI,
            language="ko"
        )

        assert document.chunk_count > 0
        print(f"✅ Document indexed: {document.chunk_count} chunks")

        # Step 2: 질문 → 답변 생성
        print("\n" + "="*60)
        print("Step 2: 질문에 답변 생성")
        print("="*60)

        question = "FastAPI를 어떻게 설치하나요?"
        result = await query_handler.query(
            question=question,
            language="ko",
            temperature=0.0
        )

        # 검증
        assert "answer" in result
        assert "sources" in result
        assert "metadata" in result

        answer = result["answer"]
        sources = result["sources"]

        print(f"\n질문: {question}")
        print(f"\n답변:\n{answer}")
        print(f"\n출처: {len(sources)}개의 문서 청크 사용")

        # 답변에 설치 관련 내용이 포함되어야 함
        assert "pip install" in answer.lower() or "설치" in answer

        # 출처 정보 확인
        assert len(sources) > 0
        for source in sources:
            assert "document_id" in source
            assert "similarity_score" in source

        print("\n✅ RAG pipeline working correctly!")

    @pytest.mark.asyncio
    async def test_conversational_query(
        self,
        document_processor,
        query_handler,
        fastapi_docs_file
    ):
        """
        대화형 질의응답 테스트:
        1. 문서 업로드
        2. 첫 번째 질문
        3. 후속 질문 (대화 컨텍스트 유지)
        """
        # 문서 업로드
        document = await document_processor.upload_and_process(
            source_file_path=fastapi_docs_file,
            title="FastAPI Guide"
        )

        # 대화 시작
        conversation = Conversation.create(
            title="FastAPI 질문",
            document_ids=[document.id]
        )

        # 첫 번째 질문
        print("\n" + "="*60)
        print("첫 번째 질문")
        print("="*60)

        conversation = await query_handler.query_with_conversation(
            conversation=conversation,
            question="FastAPI의 주요 특징은 무엇인가요?",
            language="ko"
        )

        assert conversation.message_count == 2  # 질문 + 답변

        first_answer = conversation.messages[-1].content
        print(f"\n답변: {first_answer[:200]}...")

        # 답변에 특징 관련 내용이 있어야 함
        assert any(keyword in first_answer for keyword in ["성능", "빠른", "특징", "직관"])

        # 두 번째 질문 (후속)
        print("\n" + "="*60)
        print("두 번째 질문 (대화 컨텍스트 유지)")
        print("="*60)

        conversation = await query_handler.query_with_conversation(
            conversation=conversation,
            question="그럼 어떻게 서버를 실행하나요?",
            language="ko"
        )

        assert conversation.message_count == 4  # 질문2개 + 답변2개

        second_answer = conversation.messages[-1].content
        print(f"\n답변: {second_answer[:200]}...")

        # 답변에 서버 실행 관련 내용이 있어야 함
        assert "uvicorn" in second_answer.lower() or "서버" in second_answer

        print("\n✅ Conversational query working correctly!")

    @pytest.mark.asyncio
    async def test_multiple_questions(
        self,
        document_processor,
        query_handler,
        fastapi_docs_file
    ):
        """여러 질문 테스트"""
        # 문서 업로드
        await document_processor.upload_and_process(
            source_file_path=fastapi_docs_file
        )

        # 질문 리스트
        questions = [
            "FastAPI는 무엇인가요?",
            "비동기 함수를 어떻게 정의하나요?",
            "자동 API 문서는 어디서 볼 수 있나요?"
        ]

        print("\n" + "="*60)
        print("여러 질문 테스트")
        print("="*60)

        for idx, question in enumerate(questions, 1):
            print(f"\n질문 {idx}: {question}")

            result = await query_handler.query(
                question=question,
                language="ko"
            )

            assert result["answer"]
            print(f"✅ 답변 생성 ({len(result['answer'])} chars)")

        print("\n✅ Multiple questions handled successfully!")

    @pytest.mark.asyncio
    async def test_no_relevant_documents(
        self,
        document_processor,
        query_handler,
        fastapi_docs_file
    ):
        """관련 없는 질문 테스트"""
        # FastAPI 문서만 업로드
        await document_processor.upload_and_process(
            source_file_path=fastapi_docs_file
        )

        # 완전히 관련 없는 질문
        question = "React에서 useState 훅은 어떻게 사용하나요?"

        result = await query_handler.query(
            question=question,
            language="ko"
        )

        # 답변이 생성되긴 하지만, "찾을 수 없습니다" 같은 표현이 있을 수 있음
        answer = result["answer"]
        print(f"\n질문: {question}")
        print(f"답변: {answer}")

        # 답변은 생성되어야 함
        assert len(answer) > 0

        print("\n✅ Handled unrelated question gracefully!")


class TestRAGQuality:
    """RAG 품질 테스트"""

    @pytest.mark.asyncio
    async def test_answer_uses_context(
        self,
        document_processor,
        query_handler,
        fastapi_docs_file
    ):
        """답변이 제공된 문서를 실제로 사용하는지 검증"""
        # 문서 업로드
        await document_processor.upload_and_process(
            source_file_path=fastapi_docs_file
        )

        # 문서에 명확히 있는 내용 질문
        question = "FastAPI를 설치하는 pip 명령어는 무엇인가요?"

        result = await query_handler.query(
            question=question,
            language="ko"
        )

        answer = result["answer"]
        sources = result["sources"]

        # 검증
        print(f"\n질문: {question}")
        print(f"답변: {answer}")
        print(f"출처: {len(sources)}개")

        # pip install fastapi가 답변에 포함되어야 함
        assert "pip install fastapi" in answer.lower()

        # 출처가 제공되어야 함
        assert len(sources) > 0

        print("\n✅ Answer correctly uses provided context!")

    @pytest.mark.asyncio
    async def test_source_attribution(
        self,
        document_processor,
        query_handler,
        fastapi_docs_file
    ):
        """출처 정보가 정확한지 검증"""
        # 문서 업로드
        document = await document_processor.upload_and_process(
            source_file_path=fastapi_docs_file
        )

        # 질문
        result = await query_handler.query(
            question="FastAPI의 특징은?",
            language="ko"
        )

        sources = result["sources"]

        # 출처 검증
        for source in sources:
            # 문서 ID가 올바른지
            assert source["document_id"] == document.id

            # 유사도 점수가 0~1 사이인지
            assert 0.0 <= source["similarity_score"] <= 1.0

            # 미리보기 텍스트가 있는지
            assert "preview" in source
            assert len(source["preview"]) > 0

        print(f"\n✅ Source attribution is accurate!")
        scores = [f"{s['similarity_score']:.3f}" for s in sources]
        print(f"   {len(sources)} sources with scores: {scores}")
