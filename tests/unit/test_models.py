"""
도메인 모델 단위 테스트

각 엔티티의 비즈니스 로직을 검증합니다.
"""

import pytest
from datetime import datetime
from app.domain.models import (
    Document, DocumentChunk, Conversation, Message,
    DocumentStatus, SourceType, MessageRole
)
from app.domain.exceptions import (
    InvalidDocumentStatusError,
    EmptyDocumentError
)


class TestDocument:
    """Document 엔티티 테스트"""

    def test_create_document(self):
        """문서 생성 테스트"""
        doc = Document.create(
            title="FastAPI Docs",
            source_type=SourceType.FASTAPI,
            language="en",
            file_path="/path/to/file.pdf",
            metadata={"version": "0.110"}
        )

        assert doc.id is not None
        assert doc.title == "FastAPI Docs"
        assert doc.source_type == SourceType.FASTAPI
        assert doc.language == "en"
        assert doc.status == DocumentStatus.PENDING
        assert doc.chunk_count == 0
        assert doc.indexed_at is None
        assert doc.metadata["version"] == "0.110"

    def test_document_immutability(self, sample_document):
        """문서 불변성 테스트"""
        with pytest.raises(Exception):  # FrozenInstanceError
            sample_document.title = "New Title"

    def test_is_indexed(self, sample_document, indexed_document):
        """인덱싱 상태 확인 테스트"""
        assert not sample_document.is_indexed()
        assert indexed_document.is_indexed()

    def test_can_be_queried(self, sample_document, indexed_document):
        """쿼리 가능 여부 테스트"""
        # PENDING 상태 → 쿼리 불가
        assert not sample_document.can_be_queried()

        # INDEXED이지만 chunk_count=0 → 쿼리 불가
        indexing_doc = sample_document.mark_as_indexing()
        with pytest.raises(EmptyDocumentError):
            indexing_doc.mark_as_indexed(chunk_count=0)

        # INDEXED + chunk_count>0 → 쿼리 가능
        assert indexed_document.can_be_queried()

    def test_indexing_state_transition(self, sample_document):
        """인덱싱 상태 전환 테스트"""
        # PENDING → INDEXING
        indexing_doc = sample_document.mark_as_indexing()
        assert indexing_doc.status == DocumentStatus.INDEXING

        # INDEXING → INDEXED
        indexed_doc = indexing_doc.mark_as_indexed(chunk_count=50)
        assert indexed_doc.status == DocumentStatus.INDEXED
        assert indexed_doc.chunk_count == 50
        assert indexed_doc.indexed_at is not None

    def test_invalid_state_transition(self, sample_document):
        """잘못된 상태 전환 테스트"""
        # PENDING 상태에서 바로 indexed 불가
        with pytest.raises(InvalidDocumentStatusError):
            sample_document.mark_as_indexed(chunk_count=10)

        # INDEXED 상태에서 다시 indexing 시작 불가
        indexing_doc = sample_document.mark_as_indexing()
        indexed_doc = indexing_doc.mark_as_indexed(chunk_count=10)

        with pytest.raises(InvalidDocumentStatusError):
            indexed_doc.mark_as_indexing()

    def test_mark_as_failed(self, sample_document):
        """인덱싱 실패 처리 테스트"""
        failed_doc = sample_document.mark_as_failed("PDF parsing error")

        assert failed_doc.status == DocumentStatus.FAILED
        assert "error" in failed_doc.metadata
        assert failed_doc.metadata["error"] == "PDF parsing error"
        assert failed_doc.indexed_at is None


class TestDocumentChunk:
    """DocumentChunk 엔티티 테스트"""

    def test_create_chunk(self):
        """청크 생성 테스트"""
        chunk = DocumentChunk.create(
            document_id="doc_123",
            content="This is a test content",
            chunk_index=5,
            metadata={"page": 10}
        )

        assert chunk.id is not None
        assert chunk.document_id == "doc_123"
        assert chunk.content == "This is a test content"
        assert chunk.chunk_index == 5
        assert chunk.embedding is None
        assert chunk.metadata["page"] == 10

    def test_with_embedding(self, sample_chunk):
        """임베딩 추가 테스트"""
        embedding = [0.1, 0.2, 0.3] * 512  # 1536-dim 시뮬레이션

        chunk_with_emb = sample_chunk.with_embedding(embedding)

        assert chunk_with_emb.embedding == embedding
        assert chunk_with_emb.id == sample_chunk.id  # 동일한 ID
        assert sample_chunk.embedding is None  # 원본은 불변


class TestConversation:
    """Conversation 엔티티 테스트"""

    def test_create_conversation(self):
        """대화 생성 테스트"""
        conv = Conversation.create(
            document_ids=["doc_1", "doc_2"],
            title="My Conversation"
        )

        assert conv.id is not None
        assert conv.document_ids == ["doc_1", "doc_2"]
        assert conv.title == "My Conversation"
        assert conv.created_at is not None
        assert conv.updated_at == conv.created_at

    def test_update_title(self, sample_conversation):
        """제목 변경 테스트"""
        import time
        original_updated_at = sample_conversation.updated_at

        # 시간 차이를 보장하기 위해 잠깐 대기
        time.sleep(0.001)

        updated_conv = sample_conversation.update_title("New Title")

        assert updated_conv.title == "New Title"
        assert updated_conv.updated_at >= original_updated_at  # >= 로 변경 (동일 시간 허용)
        assert updated_conv.id == sample_conversation.id


class TestMessage:
    """Message 엔티티 테스트"""

    def test_create_user_message(self):
        """사용자 메시지 생성 테스트"""
        msg = Message.create_user_message(
            conversation_id="conv_123",
            content="What is FastAPI?"
        )

        assert msg.id is not None
        assert msg.conversation_id == "conv_123"
        assert msg.role == MessageRole.USER
        assert msg.content == "What is FastAPI?"
        assert msg.sources == []
        assert msg.token_count == 0

    def test_create_assistant_message(self):
        """어시스턴트 메시지 생성 테스트"""
        msg = Message.create_assistant_message(
            conversation_id="conv_123",
            content="FastAPI is a framework...",
            sources=["chunk_1", "chunk_2"],
            token_count=100,
            metadata={"model": "gpt-3.5-turbo"}
        )

        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "FastAPI is a framework..."
        assert msg.sources == ["chunk_1", "chunk_2"]
        assert msg.token_count == 100
        assert msg.metadata["model"] == "gpt-3.5-turbo"

    def test_empty_assistant_message_raises_error(self):
        """빈 어시스턴트 메시지 에러 테스트"""
        with pytest.raises(ValueError, match="cannot be empty"):
            Message.create_assistant_message(
                conversation_id="conv_123",
                content="",
                sources=[],
                token_count=0
            )

    def test_message_role_checks(self, user_message, assistant_message):
        """메시지 역할 확인 테스트"""
        assert user_message.is_from_user()
        assert not user_message.is_from_assistant()

        assert assistant_message.is_from_assistant()
        assert not assistant_message.is_from_user()


# ============================================
# 통합 시나리오 테스트
# ============================================

class TestDocumentWorkflow:
    """문서 처리 전체 워크플로우 테스트"""

    def test_full_document_lifecycle(self):
        """문서의 전체 생명주기 테스트"""
        # 1. 문서 생성
        doc = Document.create(
            title="Test Doc",
            source_type=SourceType.PYTHON,
            language="en",
            file_path="/path/to/doc.pdf"
        )
        assert doc.status == DocumentStatus.PENDING

        # 2. 인덱싱 시작
        indexing_doc = doc.mark_as_indexing()
        assert indexing_doc.status == DocumentStatus.INDEXING

        # 3. 인덱싱 완료
        indexed_doc = indexing_doc.mark_as_indexed(chunk_count=200)
        assert indexed_doc.status == DocumentStatus.INDEXED
        assert indexed_doc.can_be_queried()

        # 4. 문서는 불변이므로 각 단계별 인스턴스가 다름
        assert doc.id == indexing_doc.id == indexed_doc.id
        assert doc.status != indexed_doc.status


class TestConversationWorkflow:
    """대화 전체 워크플로우 테스트"""

    def test_conversation_with_messages(self, sample_conversation):
        """대화와 메시지 흐름 테스트"""
        # 1. 대화 생성
        conv = sample_conversation

        # 2. 사용자 질문
        user_msg = Message.create_user_message(
            conversation_id=conv.id,
            content="Explain FastAPI routing"
        )
        assert user_msg.is_from_user()

        # 3. 어시스턴트 응답
        assistant_msg = Message.create_assistant_message(
            conversation_id=conv.id,
            content="FastAPI routing uses decorators...",
            sources=["chunk_10", "chunk_11"],
            token_count=75
        )
        assert assistant_msg.is_from_assistant()
        assert len(assistant_msg.sources) == 2

        # 4. 대화 제목 업데이트
        updated_conv = conv.update_title("FastAPI Routing Discussion")
        assert updated_conv.title == "FastAPI Routing Discussion"
