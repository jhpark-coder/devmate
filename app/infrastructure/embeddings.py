"""
임베딩 클라이언트 생성

로컬 Ollama 임베딩 모델(기본값: bge-m3)을 사용합니다.
bge-m3는 한국어·영어를 같은 벡터 공간에 넣어 주기 때문에,
한국어 질문으로 영어 문서를 찾는 이 프로젝트에 맞습니다.

주의: 임베딩 모델을 바꾸면 기존 인덱스와 벡터가 호환되지 않으므로
scripts/init_index.py 로 인덱스를 다시 만들어야 합니다.
"""

from typing import List, Optional

import numpy as np
import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

from app.config import settings


def _normalize(vector: List[float]) -> List[float]:
    """벡터 길이를 1로 맞춤"""
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


class NormalizedEmbeddings(Embeddings):
    """
    임베딩 벡터를 길이 1로 정규화해서 돌려주는 래퍼

    LangChain FAISS는 거리 → 0~1 관련도 점수를 계산할 때
    벡터가 정규화돼 있다고 가정한다. Ollama 임베딩은 정규화되지 않은
    값을 주기 때문에, 그대로 쓰면 점수가 음수로 나온다.
    """

    def __init__(self, base: Embeddings):
        self.base = base

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_normalize(v) for v in self.base.embed_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return _normalize(self.base.embed_query(text))


def create_embeddings(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Embeddings:
    """
    Ollama 임베딩 클라이언트 생성

    Args:
        model: 임베딩 모델 이름 (기본값: settings.OLLAMA_EMBED_MODEL)
        base_url: Ollama 서버 URL (기본값: settings.OLLAMA_BASE_URL)
    """
    return NormalizedEmbeddings(
        OllamaEmbeddings(
            model=model or settings.OLLAMA_EMBED_MODEL,
            base_url=base_url or settings.OLLAMA_BASE_URL,
            # bge-m3는 "query: "/"passage: " 같은 접두어가 필요 없음
            embed_instruction="",
            query_instruction="",
        )
    )


def is_embedding_model_available(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> bool:
    """Ollama 서버가 떠 있고 임베딩 모델이 설치돼 있는지 확인"""
    model = model or settings.OLLAMA_EMBED_MODEL
    try:
        response = requests.get(
            f"{base_url or settings.OLLAMA_BASE_URL}/api/tags", timeout=2
        )
        response.raise_for_status()
    except Exception:
        return False

    names = [m["name"] for m in response.json().get("models", [])]
    # "bge-m3" 와 "bge-m3:latest" 를 같은 것으로 취급
    return any(name == model or name.split(":")[0] == model for name in names)
