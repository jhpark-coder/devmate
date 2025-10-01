"""
텍스트 분할기 (Text Splitter)

긴 문서를 의미 있는 청크(chunk)로 분할합니다.

DESIGN.md 청크 전략:
- 크기: 1000 토큰 (설정 가능)
- 오버랩: 200 토큰 (문맥 유지)
- 우선순위: 마크다운 헤더 > 코드 블록 > 단락 > 문장
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

from app.config import settings


class CustomTextSplitter:
    """
    커스터마이징된 텍스트 분할기

    특징:
    - 코드 블록 보존: 코드 예제가 중간에 잘리지 않도록
    - 마크다운 인식: 헤더 구조 유지
    - 적절한 오버랩: 문맥 연속성 확보
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        텍스트 분할기 초기화

        Args:
            chunk_size: 청크 크기 (기본값: config.CHUNK_SIZE)
            chunk_overlap: 오버랩 크기 (기본값: config.CHUNK_OVERLAP)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # 분할 우선순위 (위에서 아래로)
        self.separators = [
            "\n## ",      # 마크다운 H2
            "\n### ",     # 마크다운 H3
            "\n#### ",    # 마크다운 H4
            "\n```\n",    # 코드 블록 시작
            "\n```",      # 코드 블록 끝
            "\n\n\n",     # 큰 단락 구분
            "\n\n",       # 단락 구분
            "\n",         # 줄바꿈
            ". ",         # 문장 끝
            " ",          # 단어
            ""            # 문자 (최후 수단)
        ]

        # LangChain RecursiveCharacterTextSplitter 사용
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,  # 글자 수 기준 (토큰 근사치)
            is_separator_regex=False
        )

    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할

        Args:
            text: 원본 텍스트

        Returns:
            청크 리스트
        """
        if not text or not text.strip():
            return []

        return self.splitter.split_text(text)

    def split_documents(
        self,
        documents: List[LangChainDocument]
    ) -> List[LangChainDocument]:
        """
        LangChain Document 리스트를 청크로 분할

        Args:
            documents: 원본 Document 리스트

        Returns:
            분할된 Document 리스트 (메타데이터 유지)
        """
        return self.splitter.split_documents(documents)

    def estimate_token_count(self, text: str) -> int:
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


class DocumentLoader:
    """
    다양한 형식의 문서를 로드

    지원 형식:
    - PDF (.pdf)
    - Text (.txt, .md)
    - Word (.docx) - Phase 2
    """

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        PDF 파일에서 텍스트 추출

        Process:
        1. pypdf로 시도 (빠름)
        2. 실패 시 pdfplumber로 재시도 (정확함)

        Args:
            file_path: PDF 파일 경로

        Returns:
            추출된 텍스트

        Raises:
            ValueError: PDF 읽기 실패 시
        """
        from pypdf import PdfReader
        import pdfplumber

        try:
            # 방법 1: pypdf (빠름)
            reader = PdfReader(file_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text = "\n\n".join(text_parts)

            # 텍스트가 거의 없으면 pdfplumber로 재시도
            if len(text.strip()) < 100:
                raise ValueError("Text too short, trying pdfplumber")

            print(f"✅ Extracted {len(text)} chars from {len(reader.pages)} pages (pypdf)")
            return text

        except Exception as e:
            print(f"⚠️ pypdf failed: {e}, trying pdfplumber...")

            # 방법 2: pdfplumber (정확함)
            try:
                with pdfplumber.open(file_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

                    text = "\n\n".join(text_parts)
                    print(f"✅ Extracted {len(text)} chars from {len(pdf.pages)} pages (pdfplumber)")
                    return text

            except Exception as e2:
                raise ValueError(f"Failed to read PDF: {e2}")

    @staticmethod
    def load_text(file_path: str, encoding: str = 'utf-8') -> str:
        """
        텍스트 파일 로드

        Args:
            file_path: 파일 경로
            encoding: 인코딩 (기본값: utf-8)

        Returns:
            파일 내용
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            print(f"✅ Loaded {len(text)} chars from text file")
            return text
        except UnicodeDecodeError:
            # UTF-8 실패 시 CP949 시도 (한국어 Windows)
            with open(file_path, 'r', encoding='cp949') as f:
                text = f.read()
            print(f"✅ Loaded {len(text)} chars from text file (CP949)")
            return text

    @staticmethod
    def load_docx(file_path: str) -> str:
        """
        Word 문서 로드 (Phase 2)

        Args:
            file_path: .docx 파일 경로

        Returns:
            추출된 텍스트
        """
        from docx import Document

        doc = Document(file_path)
        text_parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        text = "\n\n".join(text_parts)
        print(f"✅ Loaded {len(text)} chars from Word document")
        return text

    @staticmethod
    def load_file(file_path: str) -> str:
        """
        파일 확장자에 따라 자동으로 로더 선택

        Args:
            file_path: 파일 경로

        Returns:
            추출된 텍스트

        Raises:
            ValueError: 지원하지 않는 형식
        """
        file_path_lower = file_path.lower()

        if file_path_lower.endswith('.pdf'):
            return DocumentLoader.load_pdf(file_path)
        elif file_path_lower.endswith(('.txt', '.md')):
            return DocumentLoader.load_text(file_path)
        elif file_path_lower.endswith('.docx'):
            return DocumentLoader.load_docx(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path}. "
                f"Supported: .pdf, .txt, .md, .docx"
            )
