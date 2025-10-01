"""
Pydantic 스키마 정의

API 요청/응답 및 데이터 검증을 위한 DTO(Data Transfer Object)입니다.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


# ============================================
# Request Schemas (요청)
# ============================================

class DocumentUploadRequest(BaseModel):
    """문서 업로드 요청"""
    source_type: str = Field(
        ...,
        pattern="^(fastapi|langchain|python|other)$",
        description="문서 출처 타입"
    )
    language: str = Field(
        default="en",
        pattern="^(en|ko)$",
        description="문서 언어"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="추가 메타데이터 (버전, 태그 등)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "source_type": "fastapi",
                "language": "en",
                "metadata": {
                    "version": "0.110.0",
                    "tags": ["authentication", "security"]
                }
            }
        }


class QueryOptions(BaseModel):
    """질의 옵션"""
    max_sources: int = Field(
        default=3,
        ge=1,
        le=10,
        description="반환할 최대 출처 개수"
    )
    include_code_examples: bool = Field(
        default=True,
        description="코드 예제 포함 여부"
    )
    language: str = Field(
        default="ko",
        pattern="^(en|ko)$",
        description="답변 언어"
    )


class QueryRequest(BaseModel):
    """질의 요청"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="질문 내용"
    )
    document_ids: List[str] = Field(
        ...,
        min_length=1,
        description="검색 대상 문서 ID 목록"
    )
    options: Optional[QueryOptions] = Field(
        default=None,
        description="검색 옵션"
    )

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "question": "FastAPI에서 JWT 인증 어떻게 구현해?",
                "document_ids": ["doc_abc123"],
                "options": {
                    "max_sources": 3,
                    "include_code_examples": True,
                    "language": "ko"
                }
            }
        }


class ConversationCreateRequest(BaseModel):
    """대화 생성 요청"""
    document_ids: List[str] = Field(
        ...,
        min_length=1,
        description="참고할 문서 ID 목록"
    )
    title: Optional[str] = Field(
        default=None,
        max_length=100,
        description="대화 제목 (생략 시 자동 생성)"
    )


class MessageCreateRequest(BaseModel):
    """메시지 생성 요청 (대화에 질문 추가)"""
    content: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="메시지 내용"
    )
    options: Optional[QueryOptions] = None


# ============================================
# Response Schemas (응답)
# ============================================

class DocumentResponse(BaseModel):
    """문서 응답"""
    id: str
    title: str
    source_type: str
    language: str
    status: str
    chunk_count: int
    uploaded_at: datetime
    indexed_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_abc123",
                "title": "FastAPI Documentation",
                "source_type": "fastapi",
                "language": "en",
                "status": "indexed",
                "chunk_count": 450,
                "uploaded_at": "2025-10-01T10:30:00Z",
                "indexed_at": "2025-10-01T10:30:45Z",
                "metadata": {"version": "0.110.0"}
            }
        }


class DocumentListResponse(BaseModel):
    """문서 목록 응답"""
    total: int
    documents: List[DocumentResponse]


class SourceReference(BaseModel):
    """출처 참조"""
    chunk_id: str
    document_id: str
    document_title: str
    content_preview: str = Field(
        ...,
        description="내용 미리보기 (최대 200자)"
    )
    chunk_index: int
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="유사도 점수"
    )
    metadata: dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "document_id": "doc_abc123",
                "document_title": "FastAPI Documentation",
                "content_preview": "To implement JWT authentication in FastAPI...",
                "chunk_index": 42,
                "relevance_score": 0.89,
                "metadata": {"page": 15, "section": "Security"}
            }
        }


class QueryMetadata(BaseModel):
    """쿼리 메타데이터"""
    token_count: int
    response_time_ms: int
    model: str = "gpt-3.5-turbo"
    cost_usd: Optional[float] = None


class QueryResponse(BaseModel):
    """질의 응답"""
    answer: str = Field(..., description="생성된 답변")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="참고 출처 목록"
    )
    metadata: QueryMetadata

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "FastAPI에서 JWT 인증을 구현하는 방법은...",
                "sources": [
                    {
                        "chunk_id": "chunk_001",
                        "document_title": "FastAPI Documentation",
                        "content_preview": "...",
                        "relevance_score": 0.89
                    }
                ],
                "metadata": {
                    "token_count": 450,
                    "response_time_ms": 2340,
                    "model": "gpt-3.5-turbo",
                    "cost_usd": 0.0009
                }
            }
        }


class MessageResponse(BaseModel):
    """메시지 응답"""
    id: str
    conversation_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    sources: List[SourceReference] = Field(default_factory=list)
    timestamp: datetime
    token_count: int = 0
    metadata: dict = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """대화 응답"""
    id: str
    title: str
    document_ids: List[str]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    """대화 목록 응답"""
    total: int
    conversations: List[ConversationResponse]


# ============================================
# Error Schemas (에러)
# ============================================

class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 설명")
    code: Optional[str] = Field(None, description="에러 코드")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Document not found",
                "detail": "Document with ID 'doc_xyz' does not exist",
                "code": "DOCUMENT_NOT_FOUND"
            }
        }
