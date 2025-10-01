"""
문서 프로세서 통합 테스트

실제 파일 업로드부터 인덱싱까지 전체 파이프라인을 테스트합니다.

주의:
- 실제 OpenAI API 호출 → 비용 발생 (매우 소량)
- .env 파일에 OPENAI_API_KEY 필요
"""

import pytest
import os
from pathlib import Path
import shutil
import tempfile

from langchain_openai import OpenAIEmbeddings

from app.application.document_processor import DocumentProcessor
from app.infrastructure.vector_store import FAISSVectorStore
from app.infrastructure.text_splitter import CustomTextSplitter
from app.domain.models import SourceType, DocumentStatus
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
    """테스트용 텍스트 분할기 (작은 청크 크기)"""
    return CustomTextSplitter(
        chunk_size=200,  # 테스트용 작은 크기
        chunk_overlap=50
    )


@pytest.fixture
def document_processor(vector_store, text_splitter, test_documents_path):
    """테스트용 문서 프로세서"""
    # 임시 문서 저장 경로 설정
    original_docs_path = settings.DOCUMENTS_PATH
    settings.DOCUMENTS_PATH = test_documents_path

    processor = DocumentProcessor(
        vector_store=vector_store,
        text_splitter=text_splitter
    )

    yield processor

    # Restore
    settings.DOCUMENTS_PATH = original_docs_path


@pytest.fixture
def sample_text_file(tmp_path):
    """테스트용 텍스트 파일 생성"""
    file_path = tmp_path / "sample.txt"
    content = """
# FastAPI Documentation

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+.

## Key Features

1. Fast: Very high performance, on par with NodeJS and Go
2. Fast to code: Increase the speed of development
3. Fewer bugs: Reduce about 40% of human errors
4. Intuitive: Great editor support with autocompletion
5. Easy: Designed to be easy to use and learn

## Installation

```bash
pip install fastapi
pip install uvicorn[standard]
```

## First Steps

Create a file `main.py` with:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run the server:

```bash
uvicorn main:app --reload
```

## Interactive API Docs

FastAPI provides automatic interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Path Parameters

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

## Query Parameters

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

## Request Body

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return item
```
"""
    file_path.write_text(content, encoding='utf-8')
    return str(file_path)


class TestDocumentProcessor:
    """문서 프로세서 통합 테스트"""

    @pytest.mark.asyncio
    async def test_process_document_text_file(
        self,
        document_processor,
        sample_text_file
    ):
        """텍스트 파일 처리 테스트"""
        # 문서 처리
        document = await document_processor.process_document(
            file_path=sample_text_file,
            title="FastAPI Guide",
            source_type=SourceType.FASTAPI,
            language="en"
        )

        # Document 검증
        assert document.title == "FastAPI Guide"
        assert document.status == DocumentStatus.INDEXED
        assert document.chunk_count > 0
        assert document.source_type == SourceType.FASTAPI

        print(f"\n✅ Processed document: {document.chunk_count} chunks created")

    @pytest.mark.asyncio
    async def test_upload_and_process(
        self,
        document_processor,
        sample_text_file,
        test_documents_path
    ):
        """파일 업로드 + 처리 통합 테스트"""
        # 업로드 및 처리
        document = await document_processor.upload_and_process(
            source_file_path=sample_text_file,
            title="FastAPI Tutorial",
            source_type=SourceType.PYTHON,
            language="en"
        )

        # 검증
        assert document.status == DocumentStatus.INDEXED
        assert document.chunk_count > 0

        # 파일이 복사되었는지 확인
        uploaded_files = list(test_documents_path.glob("*.txt"))
        assert len(uploaded_files) == 1
        assert uploaded_files[0].exists()

        print(f"\n✅ Uploaded and processed: {document.chunk_count} chunks")

    @pytest.mark.asyncio
    async def test_upload_without_title_auto_generates(
        self,
        document_processor,
        sample_text_file
    ):
        """제목 없이 업로드 시 자동 생성 테스트"""
        document = await document_processor.upload_and_process(
            source_file_path=sample_text_file
            # title 생략 → 파일명에서 자동 생성
        )

        # 제목이 파일명(확장자 제외)으로 생성되었는지 확인
        assert document.title == "sample"

    @pytest.mark.asyncio
    async def test_search_in_processed_document(
        self,
        document_processor,
        vector_store,
        sample_text_file
    ):
        """처리된 문서에서 검색 테스트"""
        # 문서 처리
        document = await document_processor.upload_and_process(
            source_file_path=sample_text_file,
            title="FastAPI Docs"
        )

        # 검색
        results = await vector_store.similarity_search(
            query="How to install FastAPI?",
            k=2
        )

        # 검증
        assert len(results) > 0

        # 첫 번째 결과가 installation 관련이어야 함
        first_chunk, first_score = results[0]
        assert "pip install" in first_chunk.content.lower() or \
               "installation" in first_chunk.content.lower()

        print(f"\n✅ Search found {len(results)} relevant chunks")

    @pytest.mark.asyncio
    async def test_delete_document(
        self,
        document_processor,
        vector_store,
        sample_text_file
    ):
        """문서 삭제 테스트"""
        # 문서 처리
        document = await document_processor.upload_and_process(
            source_file_path=sample_text_file
        )

        # 삭제 전 검색 가능 확인
        results_before = await vector_store.similarity_search(
            query="FastAPI",
            k=5
        )
        assert len(results_before) > 0

        # 삭제
        await document_processor.delete_document(document.id)

        # 메타데이터에서 제거되었는지 확인
        assert document.id not in vector_store.chunk_metadata

        print(f"\n✅ Document deleted successfully")

    @pytest.mark.asyncio
    async def test_empty_file_raises_error(
        self,
        document_processor,
        tmp_path
    ):
        """빈 파일 처리 시 에러 테스트"""
        # 빈 파일 생성
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding='utf-8')

        # 에러 발생 확인
        with pytest.raises(ValueError, match="empty"):
            await document_processor.process_document(
                file_path=str(empty_file),
                title="Empty File",
                source_type=SourceType.OTHER
            )

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_error(
        self,
        document_processor
    ):
        """존재하지 않는 파일 업로드 시 에러 테스트"""
        with pytest.raises(FileNotFoundError):
            await document_processor.upload_and_process(
                source_file_path="/nonexistent/file.txt"
            )

    @pytest.mark.asyncio
    async def test_chunk_preview(
        self,
        document_processor
    ):
        """청크 미리보기 테스트"""
        text = """
        # Heading 1
        This is a paragraph.

        ## Heading 2
        This is another paragraph with more content.

        ### Heading 3
        Final paragraph.
        """

        chunks = document_processor.get_chunk_preview(text, max_chunks=3)

        # 검증
        assert len(chunks) <= 3
        assert all(isinstance(chunk, str) for chunk in chunks)

        print(f"\n✅ Preview generated {len(chunks)} chunks")


class TestDocumentProcessorScenarios:
    """실제 사용 시나리오 테스트"""

    @pytest.mark.asyncio
    async def test_multiple_documents_workflow(
        self,
        document_processor,
        vector_store,
        tmp_path
    ):
        """여러 문서 처리 워크플로우"""
        # 문서 1: FastAPI
        fastapi_file = tmp_path / "fastapi.txt"
        fastapi_file.write_text(
            "FastAPI is a modern web framework for Python. "
            "It uses Python type hints for validation.",
            encoding='utf-8'
        )

        # 문서 2: Django
        django_file = tmp_path / "django.txt"
        django_file.write_text(
            "Django is a high-level Python web framework. "
            "It follows the MVT (Model-View-Template) pattern.",
            encoding='utf-8'
        )

        # 두 문서 처리
        doc1 = await document_processor.upload_and_process(
            source_file_path=str(fastapi_file),
            title="FastAPI Overview"
        )
        doc2 = await document_processor.upload_and_process(
            source_file_path=str(django_file),
            title="Django Overview"
        )

        # 검증
        assert doc1.status == DocumentStatus.INDEXED
        assert doc2.status == DocumentStatus.INDEXED

        # FastAPI 검색
        fastapi_results = await vector_store.similarity_search(
            "Python type hints",
            k=2
        )
        assert any("FastAPI" in chunk.content for chunk, _ in fastapi_results)

        # Django 검색
        django_results = await vector_store.similarity_search(
            "MVT pattern",
            k=2
        )
        assert any("Django" in chunk.content for chunk, _ in django_results)

        print(f"\n✅ Multiple documents processed and searchable")

    @pytest.mark.asyncio
    async def test_document_state_machine(
        self,
        document_processor,
        sample_text_file
    ):
        """문서 상태 머신 검증"""
        # PENDING 상태로 시작
        from app.domain.models import Document
        document = Document.create(
            title="Test Doc",
            source_type=SourceType.OTHER,
            language="en",
            file_path=sample_text_file
        )
        assert document.status == DocumentStatus.PENDING

        # 처리 과정에서 상태 전환 확인
        processed = await document_processor.process_document(
            file_path=sample_text_file,
            title="Test Doc",
            source_type=SourceType.OTHER
        )

        # INDEXED 상태로 전환되었는지 확인
        assert processed.status == DocumentStatus.INDEXED
        assert processed.chunk_count > 0

        print(f"\n✅ Document state machine working correctly")
