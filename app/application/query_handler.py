"""
질의응답 핸들러 (Query Handler)

사용자 질문에 대한 답변 생성 파이프라인을 관리합니다.

Process:
1. 질문 벡터화
2. 유사 문서 검색 (Retrieval)
3. Context 구성
4. LLM 프롬프트 생성
5. 답변 생성 (Generation)
6. 출처 정보 첨부
"""

from typing import List, Optional, Dict
from datetime import datetime

from app.domain.models import Conversation, Message, MessageRole, DocumentChunk
from app.infrastructure.vector_store import VectorStore
from app.infrastructure.llm_client import LLMClient, PromptBuilder
from app.config import settings


class QueryHandler:
    """
    질의응답 핸들러 유스케이스

    ARCHITECTURE.md 원칙:
    - Application Layer: 비즈니스 흐름 조율
    - Domain은 Infrastructure를 모름 (의존성 역전)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        similarity_top_k: Optional[int] = None
    ):
        """
        질의응답 핸들러 초기화

        Args:
            vector_store: 벡터 저장소 (FAISS 등)
            llm_client: LLM 클라이언트 (OpenAI 등)
            similarity_top_k: 검색할 유사 청크 수 (기본값: config.SIMILARITY_TOP_K)
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.similarity_top_k = similarity_top_k or settings.SIMILARITY_TOP_K

    async def query(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        conversation_history: Optional[List[Message]] = None,
        language: str = "ko",
        temperature: float = 0.0
    ) -> Dict:
        """
        질문에 대한 답변 생성 (RAG 파이프라인)

        Process:
        1. 벡터 검색으로 관련 문서 청크 찾기
        2. 검색된 청크를 Context로 구성
        3. 프롬프트 생성 (시스템 + 컨텍스트 + 질문)
        4. LLM으로 답변 생성
        5. 출처 정보와 함께 반환

        Args:
            question: 사용자 질문
            document_ids: 검색 대상 문서 ID 리스트 (None이면 전체 검색)
            conversation_history: 이전 대화 기록
            language: 응답 언어 (ko/en)
            temperature: LLM 생성 온도

        Returns:
            {
                "answer": str,  # 생성된 답변
                "sources": List[Dict],  # 출처 정보
                "metadata": Dict  # 메타데이터 (토큰 수 등)
            }

        Raises:
            ValueError: 질문이 비어있거나 검색 결과 없음
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        print(f"\n❓ Question: {question}")

        # 1. 벡터 검색 (Retrieval)
        print(f"\n🔍 Searching for relevant documents (top-{self.similarity_top_k})...")

        filter_dict = None
        if document_ids:
            # 특정 문서에서만 검색
            filter_dict = {"document_id": document_ids[0]} if len(document_ids) == 1 else None

        search_results = await self.vector_store.similarity_search(
            query=question,
            k=self.similarity_top_k,
            filter_dict=filter_dict
        )

        if not search_results:
            raise ValueError("No relevant documents found")

        print(f"   Found {len(search_results)} relevant chunks")

        # 2. Context 구성
        chunks: List[DocumentChunk] = []
        context_texts: List[str] = []
        sources: List[Dict] = []

        for chunk, score in search_results:
            chunks.append(chunk)
            context_texts.append(chunk.content)

            # 출처 정보 수집
            sources.append({
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "similarity_score": float(score),
                "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            })

        # 3. 프롬프트 생성
        print(f"\n📝 Building prompt...")

        system_prompt = PromptBuilder.build_rag_system_prompt(language)
        user_prompt = PromptBuilder.build_rag_user_prompt(
            question=question,
            context_chunks=context_texts,
            language=language
        )

        # 대화 히스토리 포함 (옵션)
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if conversation_history:
            # 이전 대화를 메시지 형식으로 변환
            for msg in conversation_history[-10:]:  # 최근 10개만 사용
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        messages.append({"role": "user", "content": user_prompt})

        # 토큰 수 추정
        total_prompt_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = self.llm_client.estimate_tokens(" ".join(m["content"] for m in messages))

        print(f"   Prompt: ~{estimated_tokens} tokens")

        # 4. LLM 답변 생성 (Generation)
        print(f"\n🤖 Generating answer with LLM...")

        answer = await self.llm_client.generate(
            messages=messages,
            temperature=temperature
        )

        print(f"\n✅ Answer generated ({len(answer)} chars)")

        # 5. 결과 반환
        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "question": question,
                "language": language,
                "chunks_used": len(chunks),
                "estimated_prompt_tokens": estimated_tokens,
                "answer_chars": len(answer),
                "timestamp": datetime.now().isoformat()
            }
        }

    async def query_with_conversation(
        self,
        conversation: Conversation,
        question: str,
        document_ids: Optional[List[str]] = None,
        language: str = "ko",
        temperature: float = 0.0
    ) -> Conversation:
        """
        대화 컨텍스트를 유지하며 질문

        Process:
        1. 기존 대화 히스토리 로드
        2. 질문에 답변 생성
        3. 질문과 답변을 Conversation에 추가
        4. 업데이트된 Conversation 반환

        Args:
            conversation: 기존 대화 엔티티
            question: 사용자 질문
            document_ids: 검색 대상 문서 ID
            language: 응답 언어
            temperature: LLM 생성 온도

        Returns:
            업데이트된 Conversation 엔티티

        Raises:
            ValueError: 질문이 비어있거나 검색 결과 없음
        """
        print(f"\n💬 Conversational Query (Conversation ID: {conversation.id})")

        # 1. 답변 생성
        result = await self.query(
            question=question,
            document_ids=document_ids,
            conversation_history=conversation.messages,
            language=language,
            temperature=temperature
        )

        # 2. 사용자 질문 메시지 생성
        user_message = Message.create(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=question
        )

        # 3. AI 답변 메시지 생성
        assistant_message = Message.create(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=result["answer"],
            sources=result["sources"]
        )

        # 4. Conversation에 메시지 추가
        conversation = conversation.add_message(user_message)
        conversation = conversation.add_message(assistant_message)

        print(f"✅ Conversation updated ({conversation.message_count} messages total)")

        return conversation

    async def query_streaming(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        language: str = "ko"
    ):
        """
        스트리밍 방식 답변 생성 (Phase 2)

        실시간으로 답변을 생성하며 클라이언트에 전송합니다.

        Args:
            question: 사용자 질문
            document_ids: 검색 대상 문서 ID
            language: 응답 언어

        Yields:
            답변 청크 (str)

        Note:
            Phase 2에서 구현 예정
        """
        raise NotImplementedError("Streaming is planned for Phase 2")

    def get_prompt_preview(
        self,
        question: str,
        context_chunks: List[str],
        language: str = "ko"
    ) -> str:
        """
        프롬프트 미리보기

        디버깅 및 테스트용

        Args:
            question: 질문
            context_chunks: 컨텍스트 청크 리스트
            language: 언어

        Returns:
            생성될 프롬프트 전문
        """
        system_prompt = PromptBuilder.build_rag_system_prompt(language)
        user_prompt = PromptBuilder.build_rag_user_prompt(
            question=question,
            context_chunks=context_chunks,
            language=language
        )

        return f"""[SYSTEM]
{system_prompt}

[USER]
{user_prompt}"""
