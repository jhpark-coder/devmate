"""
pytest 설정 및 공통 픽스처
"""

import pytest
from datetime import datetime
from app.domain.models import Document, DocumentChunk, Conversation, Message
from app.domain.models import DocumentStatus, SourceType, MessageRole


@pytest.fixture
def sample_document():
    """테스트용 Document 픽스처"""
    return Document.create(
        title="Test FastAPI Documentation",
        source_type=SourceType.FASTAPI,
        language="en",
        file_path="/test/path/fastapi.pdf",
        metadata={"version": "0.110.0"}
    )


@pytest.fixture
def indexed_document(sample_document):
    """인덱싱 완료된 Document 픽스처"""
    indexing_doc = sample_document.mark_as_indexing()
    return indexing_doc.mark_as_indexed(chunk_count=100)


@pytest.fixture
def sample_chunk(sample_document):
    """테스트용 DocumentChunk 픽스처"""
    return DocumentChunk.create(
        document_id=sample_document.id,
        content="FastAPI is a modern web framework for building APIs",
        chunk_index=0,
        metadata={"page": 1, "section": "Introduction"}
    )


@pytest.fixture
def sample_conversation(indexed_document):
    """테스트용 Conversation 픽스처"""
    return Conversation.create(
        document_ids=[indexed_document.id],
        title="Test Conversation"
    )


@pytest.fixture
def user_message(sample_conversation):
    """테스트용 사용자 메시지 픽스처"""
    return Message.create_user_message(
        conversation_id=sample_conversation.id,
        content="What is FastAPI?"
    )


@pytest.fixture
def assistant_message(sample_conversation):
    """테스트용 어시스턴트 메시지 픽스처"""
    return Message.create_assistant_message(
        conversation_id=sample_conversation.id,
        content="FastAPI is a modern web framework...",
        sources=["chunk_001", "chunk_002"],
        token_count=50,
        metadata={"model": "gpt-3.5-turbo"}
    )
