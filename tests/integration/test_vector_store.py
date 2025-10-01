"""
벡터 스토어 통합 테스트

실제 OpenAI API를 호출하여 FAISS 벡터 스토어를 테스트합니다.

주의:
- 실제 API 호출 → 비용 발생 (매우 소량)
- .env 파일에 OPENAI_API_KEY 필요
"""

import pytest
import os
from pathlib import Path
import shutil

from langchain_openai import OpenAIEmbeddings

from app.infrastructure.vector_store import FAISSVectorStore
from app.domain.models import DocumentChunk
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
def sample_chunks():
    """테스트용 문서 청크"""
    return [
        DocumentChunk.create(
            document_id="doc_test_001",
            content="FastAPI is a modern, fast web framework for building APIs with Python 3.7+",
            chunk_index=0,
            metadata={"page": 1, "section": "Introduction"}
        ),
        DocumentChunk.create(
            document_id="doc_test_001",
            content="FastAPI is based on standard Python type hints and provides automatic API documentation",
            chunk_index=1,
            metadata={"page": 1, "section": "Features"}
        ),
        DocumentChunk.create(
            document_id="doc_test_001",
            content="Django is a high-level Python web framework that encourages rapid development",
            chunk_index=2,
            metadata={"page": 2, "section": "Comparison"}
        ),
    ]


class TestFAISSVectorStore:
    """FAISS 벡터 스토어 통합 테스트"""

    @pytest.mark.asyncio
    async def test_add_documents(self, vector_store, sample_chunks):
        """문서 추가 테스트"""
        # 문서 추가
        await vector_store.add_documents(
            chunks=sample_chunks,
            document_id="doc_test_001"
        )

        # 메타데이터 확인
        assert "doc_test_001" in vector_store.chunk_metadata
        assert len(vector_store.chunk_metadata["doc_test_001"]) == 3

    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_store, sample_chunks):
        """유사도 검색 테스트"""
        # 문서 추가
        await vector_store.add_documents(
            chunks=sample_chunks,
            document_id="doc_test_001"
        )

        # 검색 (FastAPI 관련 질의)
        results = await vector_store.similarity_search(
            query="What is FastAPI?",
            k=2
        )

        # 검증
        assert len(results) == 2

        # 첫 번째 결과가 FastAPI 관련이어야 함
        first_chunk, first_score = results[0]
        assert "FastAPI" in first_chunk.content
        assert 0.0 <= first_score <= 1.0

        # 스코어가 내림차순이어야 함
        assert results[0][1] >= results[1][1]

    @pytest.mark.asyncio
    async def test_similarity_search_with_filter(self, vector_store, sample_chunks):
        """필터링 검색 테스트"""
        # 문서 추가
        await vector_store.add_documents(
            chunks=sample_chunks,
            document_id="doc_test_001"
        )

        # 특정 문서만 검색
        results = await vector_store.similarity_search(
            query="Python web framework",
            k=5,
            filter_dict={"document_id": "doc_test_001"}
        )

        # 모든 결과가 해당 문서에서 나와야 함
        for chunk, score in results:
            assert chunk.document_id == "doc_test_001"

    @pytest.mark.asyncio
    async def test_save_and_load_index(self, embeddings, test_index_path, sample_chunks):
        """인덱스 저장/로드 테스트"""
        # 1. 인덱스 생성 및 문서 추가
        store1 = FAISSVectorStore(
            embeddings=embeddings,
            index_path=str(test_index_path)
        )
        await store1.add_documents(sample_chunks, "doc_test_001")

        # 2. 저장
        store1.save_index()

        # 3. 새 인스턴스로 로드
        store2 = FAISSVectorStore(
            embeddings=embeddings,
            index_path=str(test_index_path)
        )

        # 4. 검색이 작동해야 함
        results = await store2.similarity_search("FastAPI", k=1)
        assert len(results) == 1
        assert "FastAPI" in results[0][0].content

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, vector_store, sample_chunks):
        """문서 삭제 테스트"""
        # 문서 추가
        await vector_store.add_documents(sample_chunks, "doc_test_001")

        # 삭제 전 검색
        results_before = await vector_store.similarity_search("FastAPI", k=5)
        assert len(results_before) > 0

        # 삭제
        await vector_store.delete_by_document_id("doc_test_001")

        # 메타데이터에서 제거되었는지 확인
        assert "doc_test_001" not in vector_store.chunk_metadata

    @pytest.mark.asyncio
    async def test_empty_chunks_raises_error(self, vector_store):
        """빈 청크 리스트 에러 테스트"""
        with pytest.raises(ValueError, match="Cannot add empty chunks"):
            await vector_store.add_documents([], "doc_test")

    @pytest.mark.asyncio
    async def test_search_empty_store_raises_error(self, vector_store):
        """빈 스토어 검색 에러 테스트"""
        with pytest.raises(ValueError, match="Vector store is empty"):
            await vector_store.similarity_search("test query")


class TestVectorStoreScenarios:
    """실제 사용 시나리오 테스트"""

    @pytest.mark.asyncio
    async def test_multiple_documents(self, vector_store):
        """여러 문서 처리 시나리오"""
        # 문서 1: FastAPI
        fastapi_chunks = [
            DocumentChunk.create(
                document_id="doc_fastapi",
                content="FastAPI uses Python type hints for validation",
                chunk_index=0
            )
        ]

        # 문서 2: Django
        django_chunks = [
            DocumentChunk.create(
                document_id="doc_django",
                content="Django uses models, views, and templates pattern",
                chunk_index=0
            )
        ]

        # 추가
        await vector_store.add_documents(fastapi_chunks, "doc_fastapi")
        await vector_store.add_documents(django_chunks, "doc_django")

        # FastAPI 검색
        fastapi_results = await vector_store.similarity_search(
            "type hints",
            k=1
        )
        assert "FastAPI" in fastapi_results[0][0].content

        # Django 검색
        django_results = await vector_store.similarity_search(
            "MVT pattern",
            k=1
        )
        assert "Django" in django_results[0][0].content

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, vector_store, sample_chunks):
        """유사도 점수 검증"""
        await vector_store.add_documents(sample_chunks, "doc_test_001")

        # 매우 관련 있는 질의
        high_relevance_results = await vector_store.similarity_search(
            "FastAPI Python web framework",
            k=1
        )

        # 덜 관련 있는 질의
        low_relevance_results = await vector_store.similarity_search(
            "machine learning algorithms",
            k=1
        )

        # 관련성 높은 질의의 점수가 더 높아야 함
        assert high_relevance_results[0][1] > low_relevance_results[0][1]
