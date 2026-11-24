"""
LLM 클라이언트 (로컬 Ollama)

로컬에서 실행되는 Ollama LLM을 감싸는 추상화 레이어입니다.
외부 API 키 없이 동작합니다.

ARCHITECTURE.md 원칙:
- Infrastructure Layer: 외부 시스템 통신
- Domain은 LLM 구현을 모름 (의존성 역전)
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import requests
import json

from app.config import settings


class LLMClient(ABC):
    """
    LLM 클라이언트 추상 인터페이스

    다른 LLM 백엔드를 붙일 때 이 인터페이스만 구현하면 됨
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
5. **한국어 답변**: 모든 답변은 한국어로 작성하세요. 기술 용어는 영어를 괄호로 병기하세요 (예: "의존성 주입(Dependency Injection)").
6. **구조화된 답변**: 복잡한 내용은 단계별로 나누어 설명하세요.
7. **문자 규칙**:
   - 한국어는 반드시 한글(ㄱ-ㅎ, ㅏ-ㅣ, 가-힣)만 사용하세요
   - 영어는 알파벳(A-Z, a-z)만 사용하세요
   - 중국어 한자(罗, 箭, 頭 등)나 일본어, 베트남어는 절대 사용하지 마세요
   - 예시: "화살표 함수" (올바름), "아罗우 함수" (잘못됨)

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


class OllamaClient(LLMClient):
    """
    Ollama 로컬 LLM 클라이언트 구현

    로컬에서 실행되는 Ollama 모델 사용 (llama3.2, qwen, gemma 등)

    특징:
    - 완전 무료
    - 데이터 외부 유출 없음
    - 인터넷 불필요
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        Ollama 클라이언트 초기화

        Args:
            model: Ollama 모델 이름 (기본값: settings.OLLAMA_MODEL)
            base_url: Ollama 서버 URL (기본값: settings.OLLAMA_BASE_URL)
            temperature: 생성 온도 (기본값: settings.LLM_TEMPERATURE)
        """
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        대화 생성 (Ollama API 사용)

        Args:
            messages: 대화 메시지 리스트
            temperature: 생성 온도 (None일 경우 기본값 사용)
            max_tokens: 최대 토큰 수 (Ollama는 num_predict 사용)

        Returns:
            생성된 응답 텍스트

        Raises:
            ConnectionError: Ollama 서버 연결 실패
            Exception: API 호출 실패
        """
        if not messages:
            raise ValueError("Messages cannot be empty")

        # 온도 설정
        temp = temperature if temperature is not None else self.temperature

        try:
            # Ollama API 호출
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        # 상한이 없으면(-1) 작은 모델이 같은 말을 반복하며 끝나지 않는 경우가 있음
                        "num_predict": max_tokens or settings.LLM_MAX_TOKENS
                    }
                },
                timeout=120  # 2분 타임아웃
            )

            response.raise_for_status()

            # 응답 파싱
            result = response.json()
            return result["message"]["content"]

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Ollama 서버에 연결할 수 없습니다. "
                "Ollama가 실행 중인지 확인하세요.\n"
                "설치: https://ollama.com/download\n"
                "실행: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise Exception("Ollama 응답 시간 초과 (2분)")
        except Exception as e:
            raise Exception(f"Ollama 생성 실패: {e}")

    def estimate_tokens(self, text: str) -> int:
        """
        대략적인 토큰 수 추정

        Args:
            text: 텍스트

        Returns:
            추정 토큰 수
        """
        # 간단한 휴리스틱
        char_count = len(text)
        return char_count // 3

    def is_available(self) -> bool:
        """
        Ollama 서버 사용 가능 여부 확인

        Returns:
            사용 가능 여부
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """
        사용 가능한 모델 목록 조회

        Returns:
            모델 이름 리스트
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            result = response.json()
            return [model["name"] for model in result.get("models", [])]

        except:
            return []

    def list_chat_models(self) -> List[str]:
        """
        답변 생성에 쓸 수 있는 모델 목록 (임베딩 전용 모델 제외)

        Returns:
            모델 이름 리스트
        """
        embedding_markers = ("embed", "bge-", "minilm", "e5-")
        return [
            name for name in self.list_models()
            if not any(marker in name.lower() for marker in embedding_markers)
        ]
