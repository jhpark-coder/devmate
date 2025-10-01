"""
벡터 스토어 추상 인터페이스 및 FAISS 구현

벡터 유사도 검색을 위한 인터페이스를 정의하고,
FAISS를 사용한 구현체를 제공합니다.

ARCHITECTURE.md 원칙:
- 추상 인터페이스로 구현체 교체 가능
- FAISS → Pinecone/ChromaDB 전환 용이
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import os
import pickle
import numpy as np
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument

from app.domain.models import DocumentChunk


class VectorStore(ABC):
    """
    벡터 저장소 추상 인터페이스

    벡터 임베딩을 저장하고 유사도 기반 검색을 제공합니다.
    구현체는 FAISS, Pinecone, ChromaDB 등이 가능합니다.
    """

    @abstractmethod
    async def add_documents(
        self,
        chunks: List[DocumentChunk],
        document_id: str
    ) -> None:
        """
        문서 청크를 벡터 DB에 추가

        Args:
            chunks: 추가할 DocumentChunk 리스트
            document_id: 문서 ID (필터링용)

        Raises:
            VectorStoreError: 추가 실패 시
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        유사도 기반 검색

        Args:
            query: 검색 질의
            k: 반환할 결과 개수
            filter_dict: 필터 조건 (예: {"document_id": "doc_123"})

        Returns:
            (DocumentChunk, relevance_score) 튜플 리스트
            relevance_score: 0.0 ~ 1.0 (1.0이 가장 유사)

        Raises:
            VectorStoreError: 검색 실패 시
        """
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> None:
        """
        특정 문서의 모든 청크 삭제

        Args:
            document_id: 삭제할 문서 ID

        Raises:
            VectorStoreError: 삭제 실패 시
        """
        pass

    @abstractmethod
    def save_index(self) -> None:
        """
        인덱스를 디스크에 저장

        Raises:
            VectorStoreError: 저장 실패 시
        """
        pass

    @abstractmethod
    def load_index(self) -> None:
        """
        디스크에서 인덱스 로드

        Raises:
            VectorStoreError: 로드 실패 시
        """
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS 기반 벡터 저장소 구현

    Facebook AI Similarity Search (FAISS)를 사용하여
    로컬 환경에서 빠른 벡터 검색을 제공합니다.

    특징:
    - 로컬 실행 (비용 $0)
    - 10만+ 벡터에서 sub-second 검색
    - 간단한 설정

    제약사항:
    - 분산 환경 지원 약함
    - 스케일아웃 어려움 (Phase 2에서 Pinecone 고려)
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        index_path: str,
        dimension: int = 1536  # OpenAI ada-002 dimension
    ):
        """
        FAISS 벡터 스토어 초기화

        Args:
            embeddings: OpenAI 임베딩 클라이언트
            index_path: 인덱스 저장 경로
            dimension: 벡터 차원 (기본값: 1536)
        """
        self.embeddings = embeddings
        self.index_path = Path(index_path)
        self.dimension = dimension

        # 저장 경로 생성
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # FAISS 인덱스
        self.vector_store: Optional[FAISS] = None

        # 청크 메타데이터 (document_id → chunk 매핑)
        self.chunk_metadata: Dict[str, List[DocumentChunk]] = {}

        # 인덱스 로드 시도
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """
        인덱스를 로드하거나 새로 생성
        """
        if self.index_path.exists():
            try:
                self.load_index()
                print(f"✅ FAISS index loaded from {self.index_path}")
            except Exception as e:
                print(f"⚠️ Failed to load index: {e}")
                print("Creating new index...")
                self._create_empty_index()
        else:
            self._create_empty_index()

    def _create_empty_index(self) -> None:
        """
        빈 FAISS 인덱스 생성
        """
        # LangChain FAISS는 빈 인덱스를 지원하지 않으므로
        # 첫 문서 추가 시 생성됨
        self.vector_store = None
        self.chunk_metadata = {}
        print("✅ Empty FAISS index initialized")

    async def add_documents(
        self,
        chunks: List[DocumentChunk],
        document_id: str
    ) -> None:
        """
        문서 청크를 FAISS 인덱스에 추가

        Process:
        1. 청크 내용을 LangChain Document로 변환
        2. OpenAI API로 임베딩 생성 (비용 발생)
        3. FAISS 인덱스에 추가
        4. 메타데이터 저장
        """
        if not chunks:
            raise ValueError("Cannot add empty chunks list")

        # DocumentChunk → LangChain Document 변환
        langchain_docs = [
            LangChainDocument(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                }
            )
            for chunk in chunks
        ]

        # FAISS에 추가
        if self.vector_store is None:
            # 첫 문서 추가 시 인덱스 생성
            self.vector_store = await FAISS.afrom_documents(
                langchain_docs,
                self.embeddings
            )
        else:
            # 기존 인덱스에 추가
            await self.vector_store.aadd_documents(langchain_docs)

        # 메타데이터 저장
        self.chunk_metadata[document_id] = chunks

        print(f"✅ Added {len(chunks)} chunks for document {document_id}")

    async def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        유사도 검색 수행

        Process:
        1. 쿼리를 임베딩으로 변환
        2. FAISS에서 k개 최근접 이웃 검색
        3. 거리를 유사도 점수로 변환
        4. DocumentChunk와 함께 반환
        """
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Add documents first.")

        # FAISS 검색 (with score)
        results = await self.vector_store.asimilarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter_dict
        )

        # LangChain Document → DocumentChunk 변환
        chunk_results = []
        for doc, score in results:
            metadata = doc.metadata

            # DocumentChunk 재구성
            chunk = DocumentChunk(
                id=metadata["chunk_id"],
                document_id=metadata["document_id"],
                content=doc.page_content,
                chunk_index=metadata["chunk_index"],
                metadata={k: v for k, v in metadata.items()
                         if k not in ["chunk_id", "document_id", "chunk_index"]}
            )

            chunk_results.append((chunk, score))

        return chunk_results

    async def delete_by_document_id(self, document_id: str) -> None:
        """
        특정 문서의 모든 청크 삭제

        Note: FAISS는 개별 삭제를 효율적으로 지원하지 않음
        현재는 전체 재구성 방식 사용 (Phase 2에서 개선 가능)
        """
        if document_id not in self.chunk_metadata:
            print(f"⚠️ Document {document_id} not found in metadata")
            return

        # 메타데이터에서 제거
        del self.chunk_metadata[document_id]

        # 전체 인덱스 재구성 (비효율적이지만 간단)
        # Phase 2에서 Pinecone으로 전환 시 개선됨
        all_chunks = []
        for chunks in self.chunk_metadata.values():
            all_chunks.extend(chunks)

        if all_chunks:
            # 재구성
            self._create_empty_index()
            for doc_id, chunks in self.chunk_metadata.items():
                await self.add_documents(chunks, doc_id)
        else:
            # 모든 문서 삭제됨
            self._create_empty_index()

        print(f"✅ Deleted document {document_id}")

    def save_index(self) -> None:
        """
        FAISS 인덱스를 디스크에 저장

        저장 내용:
        1. FAISS 인덱스 (벡터)
        2. 청크 메타데이터 (pickle)
        """
        if self.vector_store is None:
            print("⚠️ No index to save")
            return

        # FAISS 인덱스 저장
        self.vector_store.save_local(str(self.index_path))

        # 메타데이터 저장
        metadata_path = self.index_path / "chunk_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunk_metadata, f)

        print(f"✅ FAISS index saved to {self.index_path}")

    def load_index(self) -> None:
        """
        디스크에서 FAISS 인덱스 로드
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")

        # FAISS 인덱스 로드
        self.vector_store = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True  # 로컬 파일이므로 안전
        )

        # 메타데이터 로드
        metadata_path = self.index_path / "chunk_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.chunk_metadata = pickle.load(f)

        print(f"✅ FAISS index loaded from {self.index_path}")


class VectorStoreError(Exception):
    """벡터 스토어 관련 예외"""
    pass
