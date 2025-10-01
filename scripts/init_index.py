"""
초기 인덱싱 스크립트

다운로드된 공식 문서들을 벡터 DB에 인덱싱합니다.

사용법:
    python scripts/init_index.py
"""

import asyncio
from pathlib import Path
import sys

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_openai import OpenAIEmbeddings

from app.application.document_processor import DocumentProcessor
from app.infrastructure.vector_store import FAISSVectorStore
from app.infrastructure.text_splitter import CustomTextSplitter
from app.domain.models import SourceType
from app.config import settings


async def index_documents():
    """다운로드된 문서들을 인덱싱"""

    print("="*60)
    print("🚀 DevMate 초기 인덱싱 시작")
    print("="*60)

    # 1. 초기화
    print("\n📦 시스템 초기화 중...")

    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        model="text-embedding-ada-002"
    )

    # 프로덕션 인덱스 경로 사용
    index_path = settings.VECTOR_STORE_PATH / "production_index"

    vector_store = FAISSVectorStore(
        embeddings=embeddings,
        index_path=str(index_path)
    )

    text_splitter = CustomTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    document_processor = DocumentProcessor(
        vector_store=vector_store,
        text_splitter=text_splitter
    )

    print("✅ 초기화 완료!")

    # 2. 문서 소스 디렉토리
    source_docs_dir = Path("data/source_docs")

    if not source_docs_dir.exists():
        print(f"\n❌ 문서 디렉토리가 없습니다: {source_docs_dir}")
        print("   먼저 다음을 실행하세요:")
        print("   python scripts/download_docs.py --all")
        return

    # 3. 각 소스별로 문서 인덱싱
    source_mapping = {
        "fastapi": SourceType.FASTAPI,
        "langchain": SourceType.LANGCHAIN,
        "python": SourceType.PYTHON
    }

    total_documents = 0
    total_chunks = 0

    for source_name, source_type in source_mapping.items():
        source_dir = source_docs_dir / source_name

        if not source_dir.exists():
            print(f"\n⚠️  {source_name} 디렉토리가 없습니다. 건너뜁니다.")
            continue

        print(f"\n{'='*60}")
        print(f"📚 {source_name.upper()} 문서 인덱싱 시작")
        print(f"{'='*60}")

        # 디렉토리 내 모든 .txt 파일 찾기
        txt_files = list(source_dir.glob("*.txt"))

        if not txt_files:
            print(f"   ⚠️  .txt 파일이 없습니다.")
            continue

        print(f"   발견된 문서: {len(txt_files)}개")

        for txt_file in txt_files:
            try:
                # 파일명을 제목으로 사용 (확장자 제외)
                title = f"{source_name.upper()} - {txt_file.stem.replace('_', ' ').title()}"

                print(f"\n   📄 처리 중: {txt_file.name}")

                # 문서 처리
                document = await document_processor.process_document(
                    file_path=str(txt_file),
                    title=title,
                    source_type=source_type,
                    language="en"
                )

                print(f"   ✅ 완료: {document.chunk_count} chunks")

                total_documents += 1
                total_chunks += document.chunk_count

            except Exception as e:
                print(f"   ❌ 실패: {e}")
                continue

    # 4. 결과 요약
    print("\n" + "="*60)
    print("✅ 인덱싱 완료!")
    print("="*60)
    print(f"\n📊 통계:")
    print(f"   - 총 문서: {total_documents}개")
    print(f"   - 총 청크: {total_chunks}개")
    print(f"   - 인덱스 위치: {index_path.absolute()}")

    print("\n🎉 DevMate 사용 준비 완료!")
    print("\n다음 단계:")
    print("   python demo.py")


def main():
    """메인 함수"""
    asyncio.run(index_documents())


if __name__ == "__main__":
    main()
