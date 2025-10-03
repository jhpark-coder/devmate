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
from app.infrastructure.llm_client import OpenAIClient, OllamaClient
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

    # LLM 선택
    st.markdown("### 🧠 LLM 선택")

    llm_option = st.radio(
        "사용할 LLM을 선택하세요:",
        ["Llama 3.2 (무료)", "GPT-4 (유료)"],
        index=0,
        help="Llama 3.2: 무료, 로컬 실행\nGPT-4: 빠르고 정확 (유료)",
        key="llm_option"
    )

    # Ollama 상태 확인
    if "Llama" in llm_option or "무료" in llm_option:
        ollama_client = OllamaClient()
        if ollama_client.is_available():
            st.success("✅ Ollama 연결됨")
            models = ollama_client.list_models()
            if models:
                st.info(f"사용 가능한 모델: {', '.join(models)}")
        else:
            st.error("❌ Ollama 연결 실패")
            st.markdown("""
            **Ollama 설치 방법:**
            1. [ollama.com](https://ollama.com/download) 에서 다운로드
            2. 설치 후 `ollama pull llama3.2` 실행
            3. `ollama serve` 로 서버 시작
            """)

    st.markdown("---")

    st.markdown("### 📚 지원 문서")

    # 문서 구조 정의
    doc_structure = {
        "Backend": {
            "Python": {"count": 3, "icon": "🐍"},
            "FastAPI": {"count": 19, "icon": "⚡"},
            "Django": {"count": 14, "icon": "🎸"},
            "Spring Boot": {"count": 35, "icon": "☕"},
            "Node.js": {"count": 13, "icon": "🟢"},
            "NestJS": {"count": 2, "icon": "🦅"},
            "LangChain": {"count": 4, "icon": "🔗"}
        },
        "Frontend": {
            "React": {"count": 19, "icon": "⚛️"},
            "JavaScript": {"count": 2, "icon": "📜"},
            "TypeScript": {"count": 2, "icon": "📘"},
            "HTML/CSS": {"count": 2, "icon": "🎨"},
            "Tailwind CSS": {"count": 1, "icon": "💨"}
        },
        "Database & Cache": {
            "Database/SQL": {"count": 2, "icon": "🗄️"},
            "MongoDB": {"count": 2, "icon": "🍃"},
            "Redis": {"count": 2, "icon": "💎"}
        },
        "DevOps & Infrastructure": {
            "Docker": {"count": 2, "icon": "🐳"},
            "AWS": {"count": 2, "icon": "☁️"},
            "AWS 인프라": {"count": 5, "icon": "🏗️"},
            "Nginx": {"count": 2, "icon": "🌐"},
            "Git": {"count": 2, "icon": "🔀"},
            "Flutter": {"count": 2, "icon": "📱"}
        }
    }

    # 카테고리별 문서 표시
    for category, frameworks in doc_structure.items():
        with st.expander(f"**{category}** ({sum(f['count'] for f in frameworks.values())}개)"):
            for framework, info in frameworks.items():
                if st.button(
                    f"{info['icon']} {framework} ({info['count']}개)",
                    key=f"doc_{framework}",
                    use_container_width=True
                ):
                    st.session_state.selected_framework = framework
                    st.rerun()

    st.markdown("**총 140개 문서, 22개 기술**")

    st.markdown("---")

    st.markdown("### 💡 질문 예시 (클릭해서 자동 입력)")

    # FAQ 카테고리별 정리
    faqs = {
        "🐍 Python/Backend": [
            "Python에서 asyncio 사용법은?",
            "FastAPI에서 dependency injection 사용하는 방법",
            "Django 모델 생성하고 마이그레이션하는 법",
            "NestJS 컨트롤러에서 Guard 사용하는 방법"
        ],
        "☕ Java/Spring": [
            "Java에서 ArrayList와 LinkedList 차이점",
            "Spring Boot JPA 엔티티 만들고 관계 설정하기",
            "Spring Boot에서 REST API 만드는 방법"
        ],
        "⚛️ Frontend": [
            "React useState와 useEffect 훅 사용법",
            "JavaScript ES6 화살표 함수와 map 함수",
            "TypeScript 인터페이스와 타입 정의 방법",
            "Tailwind CSS로 반응형 레이아웃 만들기"
        ],
        "🗄️ Database": [
            "MySQL에서 INNER JOIN과 LEFT JOIN 차이",
            "MongoDB 집계 파이프라인 사용법",
            "Redis로 세션 관리하는 방법"
        ],
        "🐳 DevOps/인프라": [
            "Docker Compose로 여러 컨테이너 실행하기",
            "ECS Fargate로 컨테이너 배포하는 방법",
            "ALB에 HTTPS 리스너와 SSL 인증서 추가하기",
            "Route 53에서 가중치 기반 라우팅 설정하는 법",
            "VPC에서 NAT Gateway 구성하는 방법",
            "Nginx로 리버스 프록시 설정하고 로드밸런싱하기",
            "Git 브랜치 전략 (Git Flow, GitHub Flow)"
        ],
        "☁️ AWS": [
            "AWS S3에서 파일 업로드하고 다운로드하기",
            "AWS Lambda로 서버리스 함수 만들기",
            "CloudFront로 CDN 설정하고 캐시 무효화하기"
        ]
    }

    # Expander로 카테고리별 FAQ 표시
    for category, questions in faqs.items():
        with st.expander(category):
            for q in questions:
                # 버튼 클릭 시 세션 스테이트에 질문 저장
                if st.button(q, key=f"faq_{q}", use_container_width=True):
                    st.session_state.selected_question = q
                    st.rerun()

    st.markdown("---")

    if st.button("🗑️ 대화 초기화"):
        st.session_state.conversation = None
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("Made with ❤️ by Claude Code")


@st.cache_resource
def initialize_vector_store():
    """
    벡터 스토어 초기화 (캐싱)

    임베딩과 벡터 스토어만 캐싱합니다.
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

    return vector_store


def initialize_query_handler(llm_choice: str, vector_store):
    """
    쿼리 핸들러 초기화 (LLM 선택 기반)

    Args:
        llm_choice: 선택된 LLM 옵션
        vector_store: 초기화된 벡터 스토어

    Returns:
        QueryHandler 인스턴스
    """
    # LLM Client 선택
    if "Llama" in llm_choice or "무료" in llm_choice:
        llm_client = OllamaClient(model="llama3.2")
    else:
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


def get_framework_docs(framework: str) -> list:
    """프레임워크별 실제 문서 파일 목록 가져오기"""
    from pathlib import Path

    # 프레임워크 이름을 디렉토리 이름으로 매핑
    framework_dir_map = {
        "React": "react",
        "FastAPI": "fastapi",
        "Spring Boot": "springboot",
        "Django": "django",
        "Node.js": "nodejs",
        "Python": "python",
        "JavaScript": "javascript",
        "TypeScript": "typescript",
        "Database/SQL": "database",
        "MongoDB": "mongodb",
        "Redis": "redis",
        "Docker": "docker",
        "AWS": "aws",
        "AWS 인프라": "aws_infra",
        "Nginx": "nginx",
        "Git": "git",
        "Flutter": "flutter",
        "NestJS": "nestjs",
        "LangChain": "langchain",
        "HTML/CSS": "html_css",
        "Tailwind CSS": "tailwind",
        "Java": "java"
    }

    dir_name = framework_dir_map.get(framework)
    if not dir_name:
        return []

    docs_path = Path("data/source_docs") / dir_name

    if not docs_path.exists():
        return []

    # 모든 .txt 파일 목록 가져오기
    doc_files = sorted([f.stem for f in docs_path.glob("*.txt")])

    return doc_files


def format_doc_name(filename: str) -> str:
    """파일명을 한글로 번역"""
    # 파일명 → 한글 번역 매핑
    translations = {
        # React
        "hooks useState": "useState 훅",
        "hooks useEffect": "useEffect 훅",
        "react basics": "React 기초",
        "react context": "Context API",
        "react custom hooks": "커스텀 훅",
        "react error boundaries": "에러 바운더리",
        "react forms": "폼 처리",
        "react lazy suspense": "Lazy/Suspense",
        "react lifecycle": "라이프사이클",
        "react memo optimization": "메모이제이션 최적화",
        "react nextjs": "Next.js",
        "react performance": "성능 최적화",
        "react portals": "포탈",
        "react redux": "Redux 상태관리",
        "react refs": "Refs 사용법",
        "react routing": "라우팅",
        "react styled components": "Styled Components",
        "react testing": "테스팅",
        "react typescript": "TypeScript 연동",

        # FastAPI
        "dependency injection": "의존성 주입",
        "async await": "비동기 처리",
        "fastapi background tasks": "백그라운드 태스크",
        "fastapi caching": "캐싱",
        "fastapi cors": "CORS 설정",
        "fastapi database": "데이터베이스 연동",
        "fastapi error handling": "에러 처리",
        "fastapi file upload": "파일 업로드",
        "fastapi lifespan": "라이프사이클",
        "fastapi middleware": "미들웨어",
        "fastapi pagination": "페이지네이션",
        "fastapi response model": "응답 모델",
        "fastapi security": "보안",
        "fastapi testing": "테스팅",
        "fastapi websocket": "WebSocket",
        "path parameters": "경로 파라미터",
        "query parameters": "쿼리 파라미터",
        "request body": "요청 본문",
        "tutorial intro": "FastAPI 시작하기",

        # Spring Boot
        "caching redis integration": "Redis 캐싱 연동",
        "data jpa relationships": "JPA 엔티티 관계",
        "messaging kafka integration": "Kafka 메시징 연동",
        "security csrf protection": "CSRF 보호",
        "security jwt authentication": "JWT 인증",
        "security oauth2 integration": "OAuth2 연동",
        "spring aop": "AOP",
        "spring auditing": "감사(Auditing)",
        "spring batch": "배치 처리",
        "spring bean": "Bean 관리",
        "spring boot intro": "Spring Boot 시작하기",
        "spring boot jpa": "JPA 기본",
        "spring cloud": "Spring Cloud",
        "spring cors": "CORS 설정",
        "spring data redis": "Redis 데이터 연동",
        "spring datasource": "데이터소스 설정",
        "spring dependency injection": "의존성 주입",
        "spring docker": "Docker 배포",
        "spring exception handling": "예외 처리",
        "spring file upload": "파일 업로드",
        "spring kafka": "Kafka 연동",
        "spring logging": "로깅",
        "spring metrics monitoring": "메트릭 모니터링",
        "spring native query": "네이티브 쿼리",
        "spring pagination": "페이지네이션",
        "spring profiles advanced": "프로파일 고급",
        "spring properties": "Properties 설정",
        "spring query methods": "쿼리 메서드",
        "spring security": "Spring Security",
        "spring specifications": "Specifications",
        "spring swagger": "Swagger API 문서",
        "spring testing": "테스팅",
        "spring validation": "유효성 검증",
        "testing integration tests": "통합 테스트",
        "testing unit tests": "단위 테스트",

        # Django
        "django models": "모델 정의",
        "django views": "뷰 작성",
        "django forms": "폼 처리",
        "django orm": "ORM 쿼리",
        "django rest framework": "REST API",
        "django authentication": "인증 시스템",

        # Node.js
        "nodejs express": "Express 서버",
        "nodejs async": "비동기 처리",
        "nodejs streams": "Stream 처리",
        "nodejs jwt": "JWT 인증",
        "nodejs mongodb": "MongoDB 연동",

        # Python
        "python basics": "파이썬 기초",
        "python async": "비동기 처리",
        "python decorators": "데코레이터",

        # JavaScript
        "javascript es6": "ES6+ 문법",
        "javascript async": "비동기 처리",

        # TypeScript
        "typescript basics": "TypeScript 기초",
        "typescript types": "타입 시스템",

        # MongoDB
        "mongodb basics": "MongoDB 기초",
        "mongodb nodejs": "Node.js 연동",

        # Redis
        "redis basics": "Redis 기초",
        "redis caching": "캐싱 전략",

        # Docker
        "docker basics": "Docker 기초",
        "docker compose": "Docker Compose",

        # AWS
        "aws basics": "AWS 기초",
        "aws services": "AWS 서비스",

        # AWS 인프라
        "alb nlb": "로드 밸런서(ALB/NLB)",
        "cloudfront": "CloudFront CDN",
        "ecs basics": "ECS 컨테이너",
        "route53": "Route53 DNS",
        "vpc networking": "VPC 네트워킹",

        # Nginx
        "nginx basics": "Nginx 기초",
        "nginx proxy": "리버스 프록시",

        # Git
        "git basics": "Git 기초",
        "git collaboration": "협업 워크플로우",

        # Flutter
        "flutter basics": "Flutter 기초",
        "flutter state": "상태 관리",

        # NestJS
        "nestjs basics": "NestJS 기초",
        "nestjs advanced": "고급 기능",

        # LangChain
        "introduction": "LangChain 소개",
        "chains": "체인(Chains)",
        "retrieval qa": "검색 기반 QA",
        "vector stores": "벡터 스토어",

        # HTML/CSS
        "html basics": "HTML 기초",
        "css basics": "CSS 기초",

        # Tailwind CSS
        "tailwind basics": "Tailwind 기초",

        # Java
        "java basics": "Java 기초",
        "java collections": "컬렉션 프레임워크",
        "java oop": "객체지향 프로그래밍"
    }

    # 소문자로 변환하고 언더스코어를 공백으로
    key = filename.lower().replace("_", " ")

    # 매핑에서 찾기
    if key in translations:
        return translations[key]

    # 매핑에 없으면 일부 자동 변환
    name = filename.replace("_", " ")
    name = " ".join(word.capitalize() for word in name.split())
    return name


def main():
    """메인 함수"""

    # 벡터 스토어 초기화 (캐싱)
    with st.spinner("🔄 벡터 스토어 로딩 중..."):
        vector_store = initialize_vector_store()

    # 세션 상태 초기화
    initialize_session_state()

    # 사이드바에서 선택된 LLM 가져오기
    # (사이드바는 이미 위에서 렌더링됨)
    llm_option = st.session_state.get('llm_option', 'Llama 3.2 (무료)')

    # 쿼리 핸들러 초기화 (선택된 LLM 사용)
    query_handler = initialize_query_handler(llm_option, vector_store)

    # 메인 타이틀
    st.title("🤖 DevMate - 개발자 문서 AI 어시스턴트")
    st.markdown("**140개 문서로 학습된 AI 어시스턴트** - Python, FastAPI, Spring Boot, React, Node.js, Django 등 22개 주요 기술 스택 공식 문서 기반")

    st.markdown("---")

    # 선택된 프레임워크가 있으면 문서 목록 표시
    if st.session_state.get('selected_framework'):
        framework = st.session_state.selected_framework

        # 프레임워크 아이콘 매핑
        framework_icons = {
            "React": "⚛️", "FastAPI": "⚡", "Django": "🎸", "Spring Boot": "☕",
            "Node.js": "🟢", "Python": "🐍", "JavaScript": "📜", "TypeScript": "📘",
            "Database/SQL": "🗄️", "MongoDB": "🍃", "Redis": "💎", "Docker": "🐳",
            "AWS": "☁️", "AWS 인프라": "🏗️", "Nginx": "🌐", "Git": "🔀",
            "Flutter": "📱", "NestJS": "🦅", "LangChain": "🔗", "HTML/CSS": "🎨",
            "Tailwind CSS": "💨", "Java": "☕"
        }
        icon = framework_icons.get(framework, "📚")

        # 실제 문서 목록 가져오기
        docs = get_framework_docs(framework)

        st.info(f"### {icon} {framework} - 문서 목록 ({len(docs)}개)")

        if docs:
            st.markdown("**질문하고 싶은 문서를 선택하세요:**")

            # 1열 레이아웃으로 모든 문서 표시
            for idx, doc_file in enumerate(docs):
                doc_name = format_doc_name(doc_file)
                # 질문 자동 생성
                question = f"{framework} {doc_name} 사용법"

                if st.button(
                    f"📄 {doc_name}",
                    key=f"doc_{framework}_{idx}",
                    use_container_width=True
                ):
                    st.session_state.selected_question = question
                    st.session_state.selected_framework = None
                    st.rerun()
        else:
            st.markdown("**문서를 찾을 수 없습니다. 직접 질문을 입력해보세요!**")

        if st.button("← 돌아가기", key="back_to_sidebar"):
            st.session_state.selected_framework = None
            st.rerun()

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

    # FAQ에서 선택된 질문이 있으면 자동 처리
    selected_question = st.session_state.get('selected_question', None)

    # 사용자 입력
    question = st.chat_input("질문을 입력하세요...")

    # FAQ 선택 질문이 있으면 우선 처리
    if selected_question:
        question = selected_question
        st.session_state.selected_question = None  # 처리 후 초기화

    if question:

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
        st.metric("지원 문서", "140개 (22개 기술)")


if __name__ == "__main__":
    main()
