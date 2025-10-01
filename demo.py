"""
DevMate 데모 스크립트

실제로 문서를 업로드하고 질문에 답변받을 수 있습니다.

사용법:
    python demo.py
"""

import asyncio
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from app.application.document_processor import DocumentProcessor
from app.application.query_handler import QueryHandler
from app.infrastructure.vector_store import FAISSVectorStore
from app.infrastructure.llm_client import OpenAIClient
from app.infrastructure.text_splitter import CustomTextSplitter
from app.domain.models import SourceType, Conversation
from app.config import settings


async def main():
    """메인 데모 함수"""

    print("="*60)
    print("🤖 DevMate - 개발자 문서 어시스턴트")
    print("="*60)

    # 1. 초기화
    print("\n📦 시스템 초기화 중...")

    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        model="text-embedding-ada-002"
    )

    # 프로덕션 인덱스 사용 (init_index.py로 생성된 인덱스)
    vector_store = FAISSVectorStore(
        embeddings=embeddings,
        index_path=str(settings.VECTOR_STORE_PATH / "production_index")
    )

    text_splitter = CustomTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    document_processor = DocumentProcessor(
        vector_store=vector_store,
        text_splitter=text_splitter
    )

    llm_client = OpenAIClient()

    query_handler = QueryHandler(
        vector_store=vector_store,
        llm_client=llm_client,
        similarity_top_k=3
    )

    print("✅ 초기화 완료!")

    # 2. 인덱스 확인
    print("\n" + "="*60)
    print("📚 문서 인덱스 확인")
    print("="*60)

    index_path = settings.VECTOR_STORE_PATH / "production_index"

    if not index_path.exists():
        print(f"\n❌ 인덱스가 없습니다!")
        print(f"\n다음 단계를 먼저 실행하세요:")
        print(f"   1. python scripts/download_docs.py --all")
        print(f"   2. python scripts/init_index.py")
        return

    print(f"\n✅ 인덱스 발견: {index_path}")
    print(f"   FastAPI, LangChain, Python 공식 문서가 준비되어 있습니다.")

    # 추가 문서 업로드 (선택사항)
    upload_choice = input("\n추가 문서를 업로드하시겠습니까? (y/n): ").lower()

    if upload_choice == 'y':
        file_path = input("파일 경로를 입력하세요 (.txt, .pdf, .docx): ").strip()

        if not Path(file_path).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        else:
            title = input("문서 제목을 입력하세요 (Enter: 자동): ").strip() or None

            print("\n📤 문서 업로드 중...")

            try:
                document = await document_processor.upload_and_process(
                    source_file_path=file_path,
                    title=title,
                    source_type=SourceType.OTHER,
                    language="ko"
                )

                print(f"\n✅ 문서 업로드 완료!")
                print(f"   - 제목: {document.title}")
                print(f"   - 청크 수: {document.chunk_count}")
                print(f"   - 문서 ID: {document.id}")

            except Exception as e:
                print(f"❌ 업로드 실패: {e}")

    # 3. 질의응답 루프
    print("\n" + "="*60)
    print("💬 질의응답 시작")
    print("="*60)
    print("\n사용법:")
    print("  - 질문을 입력하고 Enter를 누르세요")
    print("  - 'quit' 또는 'exit'를 입력하면 종료됩니다")
    print("  - 'new'를 입력하면 새로운 대화를 시작합니다")

    # 대화 시작
    conversation = Conversation.create(
        title="Demo Conversation",
        document_ids=[]
    )

    while True:
        print("\n" + "-"*60)
        question = input("\n❓ 질문: ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit', '종료']:
            print("\n👋 DevMate를 종료합니다. 감사합니다!")
            break

        if question.lower() == 'new':
            conversation = Conversation.create(
                title="New Conversation",
                document_ids=[]
            )
            print("\n✨ 새로운 대화를 시작합니다.")
            continue

        try:
            print("\n🤔 답변 생성 중...")

            # 답변 생성
            conversation = await query_handler.query_with_conversation(
                conversation=conversation,
                question=question,
                language="ko",
                temperature=0.0
            )

            # 마지막 답변 가져오기
            last_message = conversation.messages[-1]
            answer = last_message.content
            sources = last_message.sources

            print("\n" + "="*60)
            print("🤖 답변:")
            print("="*60)
            print(f"\n{answer}\n")

            if sources:
                print("-"*60)
                print(f"📚 출처: {len(sources)}개의 문서 청크 사용")
                for idx, source in enumerate(sources, 1):
                    score = source.get('similarity_score', 0)
                    print(f"   {idx}. 유사도: {score:.3f}")

            print("\n💬 대화 내역: {}/{} 메시지".format(
                conversation.message_count,
                conversation.message_count
            ))

        except ValueError as e:
            print(f"\n❌ 오류: {e}")
            print("   (문서가 인덱싱되지 않았거나 관련 내용이 없습니다)")
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류: {e}")


if __name__ == "__main__":
    # 비동기 함수 실행
    asyncio.run(main())
