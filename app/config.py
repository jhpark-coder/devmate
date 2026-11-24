"""
애플리케이션 설정 관리

환경 변수를 로드하고 전역 설정을 제공합니다.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Settings:
    """애플리케이션 설정"""

    # ============================================
    # 로컬 LLM (Ollama) — 외부 API 키 없이 동작
    # ============================================
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # 답변 생성 모델 (UI에서는 설치된 모델 중 고를 수 있음)
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    # 검색용 임베딩 모델 — 바꾸면 인덱스를 다시 만들어야 함
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    # 답변 최대 길이 (토큰)
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))

    # ============================================
    # Paths
    # ============================================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCUMENTS_PATH: Path = DATA_DIR / "documents"
    VECTOR_STORE_PATH: Path = DATA_DIR / "vector_store"
    DATABASE_PATH: Path = DATA_DIR / "database"
    LOGS_PATH: Path = PROJECT_ROOT / "logs"

    # ============================================
    # Vector Search Settings
    # ============================================
    SIMILARITY_TOP_K: int = int(os.getenv("SIMILARITY_TOP_K", "3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ============================================
    # Database
    # ============================================
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{DATABASE_PATH / 'devmate.db'}"
    )

    # ============================================
    # Server
    # ============================================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # ============================================
    # Monitoring
    # ============================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_COST_TRACKING: bool = os.getenv("ENABLE_COST_TRACKING", "True").lower() == "true"
    ENABLE_PERFORMANCE_METRICS: bool = os.getenv("ENABLE_PERFORMANCE_METRICS", "True").lower() == "true"

    @classmethod
    def create_directories(cls) -> None:
        """필요한 디렉토리 생성"""
        cls.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        cls.DATABASE_PATH.mkdir(parents=True, exist_ok=True)
        cls.LOGS_PATH.mkdir(parents=True, exist_ok=True)



# 전역 설정 인스턴스
settings = Settings()

# 디렉토리 생성
settings.create_directories()
