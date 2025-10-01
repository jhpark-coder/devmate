"""
LLM 클라이언트 (OpenAI Wrapper)

OpenAI API를 감싸는 추상화 레이어입니다.

ARCHITECTURE.md 원칙:
- Infrastructure Layer: 외부 시스템 통신
- Domain은 LLM 구현을 모름 (의존성 역전)
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.config import settings


class LLMClient(ABC):
    """
    LLM 클라이언트 추상 인터페이스

    추후 OpenAI → Anthropic/Gemini 등으로 교체 가능
    """

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        대화 생성

        Args:
            messages: 대화 메시지 리스트 [{"role": "system|user|assistant", "content": "..."}]
            temperature: 생성 온도 (0.0 = 결정적, 1.0 = 창의적)
            max_tokens: 최대 토큰 수

        Returns:
            생성된 응답 텍스트
        """
        pass


class OpenAIClient(LLMClient):
    """
    OpenAI LLM 클라이언트 구현

    사용 모델:
    - gpt-3.5-turbo (기본, 빠름, 저렴)
    - gpt-4 (옵션, 정확함, 비용 30배)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        OpenAI 클라이언트 초기화

        Args:
            api_key: OpenAI API 키 (기본값: config.OPENAI_API_KEY)
            model: 모델 이름 (기본값: config.OPENAI_MODEL)
            temperature: 기본 온도 (기본값: config.OPENAI_TEMPERATURE)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.default_temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE

        # LangChain ChatOpenAI 클라이언트
        self.client = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.default_temperature
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        대화 생성

        Args:
            messages: 대화 메시지 리스트
            temperature: 생성 온도 (None일 경우 기본값 사용)
            max_tokens: 최대 토큰 수

        Returns:
            생성된 응답 텍스트

        Raises:
            ValueError: 메시지 형식 오류
            Exception: API 호출 실패
        """
        if not messages:
            raise ValueError("Messages cannot be empty")

        # LangChain 메시지 형식으로 변환
        langchain_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if not role or not content:
                raise ValueError(f"Invalid message format: {msg}")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown role: {role}")

        # 파라미터 설정
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # API 호출
        try:
            # LangChain의 ainvoke 사용 (async)
            response = await self.client.ainvoke(
                langchain_messages,
                **kwargs
            )

            return response.content

        except Exception as e:
            raise Exception(f"LLM generation failed: {e}")

    def estimate_tokens(self, text: str) -> int:
        """
        대략적인 토큰 수 추정

        OpenAI 토큰화 규칙:
        - 영어: ~4 글자 = 1 토큰
        - 한국어: ~1.5 글자 = 1 토큰

        Args:
            text: 텍스트

        Returns:
            추정 토큰 수
        """
        # 간단한 휴리스틱 (실제론 tiktoken 사용해야 정확)
        char_count = len(text)

        # 영어/한국어 혼합 가정 (평균 2.5 글자 = 1 토큰)
        return char_count // 3


class PromptBuilder:
    """
    프롬프트 빌더

    RAG 시스템용 프롬프트를 구조화하여 생성합니다.
    """

    @staticmethod
    def build_rag_system_prompt(language: str = "ko") -> str:
        """
        RAG 시스템 프롬프트 생성

        Args:
            language: 응답 언어 (ko/en)

        Returns:
            시스템 프롬프트
        """
        if language == "ko":
            return """당신은 개발자를 돕는 기술 문서 어시스턴트 DevMate입니다.

**역할**:
- 제공된 문서(Context)를 기반으로 정확한 답변 제공
- 코드 예제와 설명을 명확하게 전달
- 출처를 명시하여 신뢰성 확보

**지침**:
1. **Context 기반 답변**: 제공된 문서 내용만을 사용하여 답변하세요.
2. **출처 명시**: 답변에 사용한 문서 출처를 명확히 밝히세요.
3. **모르면 인정**: Context에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
4. **코드 강조**: 코드 예제는 마크다운 코드 블록(```)으로 감싸세요.
5. **한국어 + 영어**: 기술 용어는 영어 원문을 병기하세요 (예: "비동기(Async)", "라우팅(Routing)").
6. **구조화된 답변**: 복잡한 내용은 단계별로 나누어 설명하세요.

**예시**:
질문: FastAPI에서 비동기 함수를 어떻게 정의하나요?

답변:
FastAPI에서 비동기(Async) 함수는 `async def`로 정의합니다:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

출처: FastAPI 공식 문서 - First Steps
"""
        else:  # English
            return """You are DevMate, a technical documentation assistant for developers.

**Role**:
- Provide accurate answers based on the given Context
- Clearly explain code examples
- Cite sources to ensure credibility

**Guidelines**:
1. **Context-based answers**: Use only the provided documentation.
2. **Cite sources**: Clearly indicate document sources used in your answer.
3. **Admit unknowns**: If the Context doesn't contain the information, respond with "I couldn't find that information in the provided documents."
4. **Highlight code**: Wrap code examples in markdown code blocks (```).
5. **Structured answers**: Break down complex topics into steps.

**Example**:
Question: How do I define async functions in FastAPI?

Answer:
In FastAPI, async functions are defined with `async def`:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

Source: FastAPI Official Docs - First Steps
"""

    @staticmethod
    def build_rag_user_prompt(
        question: str,
        context_chunks: List[str],
        language: str = "ko"
    ) -> str:
        """
        RAG 사용자 프롬프트 생성

        Args:
            question: 사용자 질문
            context_chunks: 검색된 문서 청크 리스트
            language: 응답 언어

        Returns:
            사용자 프롬프트
        """
        # Context 조합
        context = "\n\n---\n\n".join(context_chunks)

        if language == "ko":
            return f"""**Context** (참고 문서):
{context}

---

**질문**: {question}

위 Context를 참고하여 질문에 답변해주세요."""
        else:
            return f"""**Context** (Reference documents):
{context}

---

**Question**: {question}

Please answer the question based on the Context above."""

    @staticmethod
    def build_conversational_prompt(
        question: str,
        conversation_history: List[Dict[str, str]],
        language: str = "ko"
    ) -> List[Dict[str, str]]:
        """
        대화 히스토리를 포함한 프롬프트 생성

        Args:
            question: 현재 질문
            conversation_history: 이전 대화 기록 [{"role": "user|assistant", "content": "..."}]
            language: 응답 언어

        Returns:
            전체 메시지 리스트
        """
        messages = [
            {"role": "system", "content": PromptBuilder.build_rag_system_prompt(language)}
        ]

        # 이전 대화 추가
        messages.extend(conversation_history)

        # 현재 질문 추가
        messages.append({"role": "user", "content": question})

        return messages
