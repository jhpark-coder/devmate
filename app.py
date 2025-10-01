"""
DevMate Streamlit 웹 UI

공식 문서 기반 AI 개발자 어시스턴트 웹 인터페이스

실행 방법:
    streamlit run app.py
"""

import streamlit as st
import asyncio
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from app.application.document_processor import DocumentProcessor
from app.application.query_handler import QueryHandler
from app.infrastructure.vector_store import FAISSVectorStore
from app.infrastructure.llm_client import OpenAIClient
from app.infrastructure.text_splitter import CustomTextSplitter
from app.domain.models import Conversation, SourceType
from app.config import settings


# 페이지 설정
st.set_page_config(
    page_title="DevMate - 개발자 문서 AI 어시스턴트",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바
with st.sidebar:
    st.title("🤖 DevMate")
    st.markdown("### 개발자 문서 AI 어시스턴트")

    st.markdown("---")

    st.markdown("### 📚 지원 문서")
    st.markdown("""
    - ✅ **FastAPI** (6개 문서)
    - ✅ **LangChain** (4개 문서)
    - ✅ **Python** (3개 문서)
    """)

    st.markdown("---")

    st.markdown("### 💡 질문 예시")
    st.markdown("""
    - FastAPI에서 async 함수는?
    - LangChain vector store 사용법?
    - Python dataclass란?
    - dependency injection 예제?
    """)

    st.markdown("---")

    if st.button("🗑️ 대화 초기화"):
        st.session_state.conversation = None
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("Made with ❤️ by Claude Code")


@st.cache_resource
def initialize_system():
    """
    시스템 초기화 (캐싱)

    앱 실행 시 한 번만 초기화됩니다.
    """
    # OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        model="text-embedding-ada-002"
    )

    # FAISS Vector Store (프로덕션 인덱스)
    index_path = settings.VECTOR_STORE_PATH / "production_index"

    if not index_path.exists():
        st.error("❌ 인덱스가 없습니다!")
        st.info("""
        다음 명령어를 먼저 실행하세요:
        ```bash
        python scripts/download_docs.py
        ./venv/Scripts/python scripts/init_index.py
        ```
        """)
        st.stop()

    vector_store = FAISSVectorStore(
        embeddings=embeddings,
        index_path=str(index_path)
    )

    # LLM Client
    llm_client = OpenAIClient()

    # Query Handler
    query_handler = QueryHandler(
        vector_store=vector_store,
        llm_client=llm_client,
        similarity_top_k=3
    )

    return query_handler


def initialize_session_state():
    """세션 상태 초기화"""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


async def process_query(query_handler, question):
    """
    질문 처리

    Args:
        query_handler: QueryHandler 인스턴스
        question: 사용자 질문

    Returns:
        (answer, sources, conversation)
    """
    # 대화가 없으면 새로 생성
    if st.session_state.conversation is None:
        st.session_state.conversation = Conversation.create(
            title="Web Conversation",
            document_ids=[]
        )

    # 질문 처리
    conversation = await query_handler.query_with_conversation(
        conversation=st.session_state.conversation,
        question=question,
        language="ko",
        temperature=0.0
    )

    # 세션 상태 업데이트
    st.session_state.conversation = conversation

    # 마지막 답변 추출
    last_message = conversation.messages[-1]
    answer = last_message.content
    sources = last_message.sources

    return answer, sources, conversation


def main():
    """메인 함수"""

    # 시스템 초기화
    with st.spinner("🔄 시스템 초기화 중..."):
        query_handler = initialize_system()

    # 세션 상태 초기화
    initialize_session_state()

    # 메인 타이틀
    st.title("🤖 DevMate - 개발자 문서 AI 어시스턴트")
    st.markdown("FastAPI, LangChain, Python 공식 문서를 기반으로 답변합니다.")

    st.markdown("---")

    # 대화 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 출처 정보 표시 (assistant 메시지만)
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 출처 보기"):
                    for idx, source in enumerate(message["sources"], 1):
                        score = source.get("similarity_score", 0)
                        preview = source.get("preview", "")
                        st.markdown(f"**{idx}. 유사도: {score:.3f}**")
                        st.markdown(f"> {preview[:150]}...")
                        st.markdown("---")

    # 사용자 입력
    if question := st.chat_input("질문을 입력하세요..."):

        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(question)

        # 메시지 히스토리에 추가
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("🤔 답변 생성 중..."):
                try:
                    # asyncio 이벤트 루프 실행
                    answer, sources, conversation = asyncio.run(
                        process_query(query_handler, question)
                    )

                    # 답변 표시
                    st.markdown(answer)

                    # 출처 표시
                    if sources:
                        with st.expander(f"📚 출처 보기 ({len(sources)}개)"):
                            for idx, source in enumerate(sources, 1):
                                score = source.get("similarity_score", 0)
                                preview = source.get("preview", "")
                                st.markdown(f"**{idx}. 유사도: {score:.3f}**")
                                st.markdown(f"> {preview[:150]}...")
                                st.markdown("---")

                    # 메시지 히스토리에 추가
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except ValueError as e:
                    st.error(f"❌ 오류: {e}")
                    st.info("문서에서 관련 내용을 찾을 수 없습니다.")

                except Exception as e:
                    st.error(f"❌ 예상치 못한 오류: {e}")

    # 하단 정보
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("대화 메시지", len(st.session_state.messages))

    with col2:
        if st.session_state.conversation:
            st.metric("질문 수", st.session_state.conversation.message_count // 2)
        else:
            st.metric("질문 수", 0)

    with col3:
        st.metric("지원 문서", "13개")


if __name__ == "__main__":
    main()
