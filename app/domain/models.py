"""
도메인 모델 정의

핵심 비즈니스 엔티티를 표현합니다.
불변성 원칙을 따르며, 외부 의존성이 없습니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List
from uuid import uuid4

from .exceptions import (
    DocumentNotIndexedError,
    InvalidDocumentStatusError,
    EmptyDocumentError,
    InvalidMessageRoleError
)


class DocumentStatus(Enum):
    """문서 처리 상태"""
    PENDING = "pending"      # 업로드됨, 인덱싱 대기 중
    INDEXING = "indexing"    # 인덱싱 진행 중
    INDEXED = "indexed"      # 인덱싱 완료
    FAILED = "failed"        # 인덱싱 실패


class SourceType(Enum):
    """문서 출처 타입"""
    FASTAPI = "fastapi"
    LANGCHAIN = "langchain"
    PYTHON = "python"
    OTHER = "other"


class MessageRole(Enum):
    """메시지 역할"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class Document:
    """
    문서 엔티티

    업로드된 문서의 메타데이터를 표현합니다.
    불변 객체로 설계되어 있습니다.
    """
    id: str
    title: str
    source_type: SourceType
    language: str  # "en" | "ko"
    file_path: str
    uploaded_at: datetime
    indexed_at: Optional[datetime]
    chunk_count: int
    status: DocumentStatus
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def create(
        title: str,
        source_type: SourceType,
        language: str,
        file_path: str,
        metadata: Optional[dict] = None
    ) -> "Document":
        """
        새 문서 생성 (팩토리 메서드)

        Args:
            title: 문서 제목
            source_type: 출처 타입
            language: 언어 코드
            file_path: 파일 저장 경로
            metadata: 추가 메타데이터

        Returns:
            새로운 Document 인스턴스
        """
        return Document(
            id=str(uuid4()),
            title=title,
            source_type=source_type,
            language=language,
            file_path=file_path,
            uploaded_at=datetime.now(),
            indexed_at=None,
            chunk_count=0,
            status=DocumentStatus.PENDING,
            metadata=metadata or {}
        )

    def is_indexed(self) -> bool:
        """인덱싱 완료 여부"""
        return self.status == DocumentStatus.INDEXED

    def can_be_queried(self) -> bool:
        """
        쿼리 가능 여부

        인덱싱이 완료되고 최소 1개 이상의 청크가 있어야 합니다.
        """
        return self.is_indexed() and self.chunk_count > 0

    def mark_as_indexing(self) -> "Document":
        """
        인덱싱 시작 상태로 변경

        Returns:
            상태가 변경된 새 Document 인스턴스

        Raises:
            InvalidDocumentStatusError: PENDING 상태가 아닐 때
        """
        if self.status != DocumentStatus.PENDING:
            raise InvalidDocumentStatusError(
                f"Cannot start indexing: current status is {self.status.value}"
            )

        return Document(
            id=self.id,
            title=self.title,
            source_type=self.source_type,
            language=self.language,
            file_path=self.file_path,
            uploaded_at=self.uploaded_at,
            indexed_at=self.indexed_at,
            chunk_count=self.chunk_count,
            status=DocumentStatus.INDEXING,
            metadata=self.metadata
        )

    def mark_as_indexed(self, chunk_count: int) -> "Document":
        """
        인덱싱 완료 상태로 변경

        Args:
            chunk_count: 생성된 청크 개수

        Returns:
            상태가 변경된 새 Document 인스턴스

        Raises:
            InvalidDocumentStatusError: INDEXING 상태가 아닐 때
            EmptyDocumentError: chunk_count가 0일 때
        """
        if self.status != DocumentStatus.INDEXING:
            raise InvalidDocumentStatusError(
                f"Cannot complete indexing: current status is {self.status.value}"
            )

        if chunk_count <= 0:
            raise EmptyDocumentError("Document must have at least one chunk")

        return Document(
            id=self.id,
            title=self.title,
            source_type=self.source_type,
            language=self.language,
            file_path=self.file_path,
            uploaded_at=self.uploaded_at,
            indexed_at=datetime.now(),
            chunk_count=chunk_count,
            status=DocumentStatus.INDEXED,
            metadata=self.metadata
        )

    def mark_as_failed(self, error_message: str) -> "Document":
        """
        인덱싱 실패 상태로 변경

        Args:
            error_message: 실패 사유

        Returns:
            상태가 변경된 새 Document 인스턴스
        """
        return Document(
            id=self.id,
            title=self.title,
            source_type=self.source_type,
            language=self.language,
            file_path=self.file_path,
            uploaded_at=self.uploaded_at,
            indexed_at=None,
            chunk_count=0,
            status=DocumentStatus.FAILED,
            metadata={**self.metadata, "error": error_message}
        )


@dataclass(frozen=True)
class DocumentChunk:
    """
    문서 청크

    분할된 문서 조각과 임베딩 벡터를 표현합니다.
    """
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @staticmethod
    def create(
        document_id: str,
        content: str,
        chunk_index: int,
        metadata: Optional[dict] = None
    ) -> "DocumentChunk":
        """청크 생성 팩토리 메서드"""
        return DocumentChunk(
            id=str(uuid4()),
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            metadata=metadata or {},
            embedding=None
        )

    def with_embedding(self, embedding: List[float]) -> "DocumentChunk":
        """
        임베딩 추가

        Args:
            embedding: 벡터 임베딩 (1536-dim)

        Returns:
            임베딩이 추가된 새 DocumentChunk 인스턴스
        """
        return DocumentChunk(
            id=self.id,
            document_id=self.document_id,
            content=self.content,
            chunk_index=self.chunk_index,
            metadata=self.metadata,
            embedding=embedding
        )


@dataclass(frozen=True)
class Conversation:
    """
    대화 엔티티

    사용자와의 대화 세션을 표현합니다.
    """
    id: str
    user_id: Optional[str]  # Phase 2에서 사용
    document_ids: List[str]
    title: str
    messages: List["Message"] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def create(
        title: str,
        document_ids: List[str],
        user_id: Optional[str] = None
    ) -> "Conversation":
        """대화 생성 팩토리 메서드"""
        now = datetime.now()
        return Conversation(
            id=str(uuid4()),
            user_id=user_id,
            document_ids=document_ids,
            title=title,
            messages=[],
            created_at=now,
            updated_at=now
        )

    def update_title(self, new_title: str) -> "Conversation":
        """제목 변경"""
        return Conversation(
            id=self.id,
            user_id=self.user_id,
            document_ids=self.document_ids,
            title=new_title,
            messages=self.messages,
            created_at=self.created_at,
            updated_at=datetime.now()
        )

    def add_message(self, message: "Message") -> "Conversation":
        """
        메시지 추가

        Args:
            message: 추가할 Message 엔티티

        Returns:
            메시지가 추가된 새 Conversation 인스턴스
        """
        new_messages = self.messages + [message]
        return Conversation(
            id=self.id,
            user_id=self.user_id,
            document_ids=self.document_ids,
            title=self.title,
            messages=new_messages,
            created_at=self.created_at,
            updated_at=datetime.now()
        )

    @property
    def message_count(self) -> int:
        """메시지 개수"""
        return len(self.messages)


@dataclass(frozen=True)
class Message:
    """
    메시지 엔티티

    대화 내 개별 메시지를 표현합니다.
    """
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    sources: List[str] = field(default_factory=list)  # chunk IDs
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def create(
        conversation_id: str,
        role: MessageRole,
        content: str,
        sources: Optional[List] = None,
        token_count: int = 0,
        metadata: Optional[dict] = None
    ) -> "Message":
        """
        메시지 생성 팩토리 메서드

        Args:
            conversation_id: 대화 ID
            role: 메시지 역할 (USER/ASSISTANT/SYSTEM)
            content: 메시지 내용
            sources: 출처 정보 (딕셔너리 리스트)
            token_count: 토큰 수
            metadata: 메타데이터

        Returns:
            Message 인스턴스
        """
        if not content:
            raise ValueError("Message content cannot be empty")

        return Message(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources or [],
            timestamp=datetime.now(),
            token_count=token_count,
            metadata=metadata or {}
        )

    @staticmethod
    def create_user_message(
        conversation_id: str,
        content: str
    ) -> "Message":
        """사용자 메시지 생성"""
        return Message.create(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content
        )

    @staticmethod
    def create_assistant_message(
        conversation_id: str,
        content: str,
        sources: List[str],
        token_count: int,
        metadata: Optional[dict] = None
    ) -> "Message":
        """어시스턴트 메시지 생성"""
        return Message.create(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=content,
            sources=sources,
            token_count=token_count,
            metadata=metadata
        )

    def is_from_user(self) -> bool:
        """사용자 메시지 여부"""
        return self.role == MessageRole.USER

    def is_from_assistant(self) -> bool:
        """어시스턴트 메시지 여부"""
        return self.role == MessageRole.ASSISTANT
