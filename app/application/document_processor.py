"""
문서 처리 유스케이스 (Document Processor)

문서 업로드부터 인덱싱까지 전체 파이프라인을 관리합니다.

Process:
1. 파일 저장
2. 텍스트 추출 (PDF/TXT/DOCX)
3. 청크 분할
4. 임베딩 생성
5. 벡터 DB 저장
6. 메타데이터 저장
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from app.domain.models import Document, DocumentChunk, DocumentStatus, SourceType
from app.infrastructure.vector_store import VectorStore
from app.infrastructure.text_splitter import CustomTextSplitter, DocumentLoader
from app.config import settings


class DocumentProcessor:
    """
    문서 처리 유스케이스

    ARCHITECTURE.md 원칙:
    - Application Layer: 비즈니스 흐름 조율
    - Domain은 Infrastructure를 모름 (의존성 역전)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        text_splitter: Optional[CustomTextSplitter] = None
    ):
        """
        문서 프로세서 초기화

        Args:
            vector_store: 벡터 저장소 (FAISS 등)
            text_splitter: 텍스트 분할기 (기본값: 새 인스턴스)
        """
        self.vector_store = vector_store
        self.text_splitter = text_splitter or CustomTextSplitter()

        # 문서 저장 디렉토리 생성
        settings.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

    async def process_document(
        self,
        file_path: str,
        title: str,
        source_type: SourceType,
        language: str = "en",
        metadata: Optional[dict] = None
    ) -> Document:
        """
        문서 처리 메인 파이프라인

        Process:
        1. Document 엔티티 생성 (PENDING 상태)
        2. 텍스트 추출
        3. 청크 분할
        4. DocumentChunk 엔티티 생성
        5. 벡터 DB에 추가 (임베딩 자동 생성)
        6. Document 상태 → INDEXED

        Args:
            file_path: 원본 파일 경로
            title: 문서 제목
            source_type: 출처 타입
            language: 언어 코드
            metadata: 추가 메타데이터

        Returns:
            처리 완료된 Document

        Raises:
            ValueError: 파일 읽기 실패
            Exception: 인덱싱 실패
        """
        # 1. Document 엔티티 생성 (PENDING)
        document = Document.create(
            title=title,
            source_type=source_type,
            language=language,
            file_path=file_path,
            metadata=metadata or {}
        )

        print(f"\n📄 Processing document: {title}")
        print(f"   Source: {file_path}")
        print(f"   Document ID: {document.id}")

        try:
            # 2. 상태 → INDEXING
            document = document.mark_as_indexing()

            # 3. 텍스트 추출
            print(f"\n📖 Extracting text...")
            text = DocumentLoader.load_file(file_path)

            if not text.strip():
                raise ValueError("Extracted text is empty")

            # 4. 청크 분할
            print(f"\n✂️  Splitting into chunks...")
            chunks_text = self.text_splitter.split_text(text)
            print(f"   Created {len(chunks_text)} chunks")

            # 5. DocumentChunk 엔티티 생성
            chunks = []
            for idx, chunk_text in enumerate(chunks_text):
                chunk = DocumentChunk.create(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=idx,
                    metadata={
                        "source_type": source_type.value,
                        "language": language,
                        "char_count": len(chunk_text)
                    }
                )
                chunks.append(chunk)

            # 6. 벡터 DB에 추가 (임베딩 자동 생성)
            print(f"\n🔢 Generating embeddings and indexing...")
            await self.vector_store.add_documents(
                chunks=chunks,
                document_id=document.id
            )

            # 7. 인덱스 저장
            self.vector_store.save_index()

            # 8. 상태 → INDEXED
            document = document.mark_as_indexed(chunk_count=len(chunks))

            print(f"\n✅ Document processed successfully!")
            print(f"   Chunks: {document.chunk_count}")
            print(f"   Status: {document.status.value}")

            return document

        except Exception as e:
            # 인덱싱 실패
            print(f"\n❌ Processing failed: {e}")
            document = document.mark_as_failed(str(e))
            raise

    async def upload_and_process(
        self,
        source_file_path: str,
        title: Optional[str] = None,
        source_type: Optional[SourceType] = None,
        language: str = "en",
        metadata: Optional[dict] = None
    ) -> Document:
        """
        파일 업로드 + 처리 통합 메서드

        Process:
        1. 파일을 data/documents/로 복사
        2. 문서 처리 파이프라인 실행

        Args:
            source_file_path: 원본 파일 경로
            title: 문서 제목 (기본값: 파일명)
            source_type: 출처 타입 (기본값: OTHER)
            language: 언어 코드
            metadata: 추가 메타데이터

        Returns:
            처리 완료된 Document

        Raises:
            FileNotFoundError: 파일 없음
            ValueError: 처리 실패
        """
        # 파일 존재 확인
        source_path = Path(source_file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_file_path}")

        # 제목 자동 생성
        if title is None:
            title = source_path.stem  # 확장자 제외한 파일명

        # 출처 타입 자동 판단
        if source_type is None:
            source_type = SourceType.OTHER

        # 파일 복사 (data/documents/에 저장)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_filename = f"{timestamp}_{source_path.name}"
        dest_path = settings.DOCUMENTS_PATH / dest_filename

        print(f"\n📂 Copying file to: {dest_path}")
        shutil.copy2(source_file_path, dest_path)

        # 문서 처리
        return await self.process_document(
            file_path=str(dest_path),
            title=title,
            source_type=source_type,
            language=language,
            metadata=metadata
        )

    async def delete_document(self, document_id: str) -> None:
        """
        문서 삭제

        Process:
        1. 벡터 DB에서 제거
        2. 파일 삭제 (옵션)

        Args:
            document_id: 삭제할 문서 ID
        """
        print(f"\n🗑️  Deleting document: {document_id}")

        # 벡터 DB에서 제거
        await self.vector_store.delete_by_document_id(document_id)

        # 인덱스 저장
        self.vector_store.save_index()

        print(f"✅ Document deleted")

    def get_chunk_preview(
        self,
        text: str,
        max_chunks: int = 3
    ) -> list[str]:
        """
        텍스트 분할 미리보기

        디버깅 및 테스트용

        Args:
            text: 원본 텍스트
            max_chunks: 표시할 최대 청크 수

        Returns:
            청크 미리보기 리스트
        """
        chunks = self.text_splitter.split_text(text)
        return chunks[:max_chunks]
