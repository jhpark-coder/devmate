# DevMate 🤖

**개발자의 공식 문서 메이트 - 영어 문서를 한국어로, 복잡한 개념을 쉽게**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

---

## 📋 프로젝트 개요

DevMate는 개발자가 기술 공식 문서를 빠르고 쉽게 이해할 수 있도록 돕는 **RAG 기반 AI 챗봇**입니다.

### 해결하는 문제
- ❌ **영어 장벽**: 공식 문서 대부분이 영어로 작성됨
- ❌ **정보 과부하**: 방대한 문서에서 필요한 정보 찾기 어려움
- ❌ **맥락 부재**: 기존 검색은 단편적 결과만 제공
- ❌ **실전 예제 부족**: 이론은 있지만 코드 예시 부족

### 우리의 솔루션
- ✅ **대화형 검색**: 맥락을 이해하고 단계별 설명 제공
- ✅ **한영 병행**: 한국어 답변 + 영문 원문 병행 표시
- ✅ **코드 중심**: 실행 가능한 예제 우선 제공
- ✅ **출처 명확**: Hallucination 방지를 위한 출처 표시

---

## 🎯 주요 기능

### Phase 1 (MVP) ✅
- [x] PDF 문서 업로드 & 자동 인덱싱
- [x] 자연어 질의응답
- [x] 출처 표시 (top-3 관련 문서)
- [x] FAISS 벡터 검색

### Phase 2 (사용성) 🔄
- [ ] 대화 히스토리 저장 & 맥락 유지
- [ ] Streamlit 웹 인터페이스
- [ ] 여러 문서 동시 검색
- [ ] SQLite 기반 데이터 관리

### Phase 3 (차별화) 🎯
- [ ] 한영 병행 답변 모드
- [ ] 코드 블록 Syntax Highlighting
- [ ] 프롬프트 최적화 (답변 품질 개선)
- [ ] 비용/성능 모니터링

### Phase 4 (배포) 🚀
- [ ] Streamlit Cloud 배포
- [ ] 완전한 문서화
- [ ] 데모 시나리오

---

## 🏗️ 기술 스택

### Core
- **LLM**: OpenAI GPT-3.5-turbo
- **Framework**: LangChain
- **Vector DB**: FAISS (페이스북)
- **Embeddings**: OpenAI text-embedding-ada-002

### Backend
- **API**: FastAPI
- **DB**: SQLite (aiosqlite)
- **ORM**: SQLAlchemy 2.0 (async)

### Frontend
- **UI**: Streamlit
- **Syntax Highlighting**: Pygments

### DevOps
- **Version Control**: Git
- **Deployment**: Streamlit Cloud
- **Monitoring**: Custom logging + metrics

---

## 📁 프로젝트 구조

```
devmate/
├── app/
│   ├── domain/              # 도메인 모델 (엔티티)
│   ├── application/         # 비즈니스 로직 (유스케이스)
│   ├── infrastructure/      # 외부 연동 (DB, LLM, 벡터 스토어)
│   ├── api/                 # REST API (FastAPI)
│   └── ui/                  # 웹 UI (Streamlit)
│
├── tests/
│   ├── unit/                # 단위 테스트
│   ├── integration/         # 통합 테스트
│   └── e2e/                 # E2E 테스트
│
├── data/
│   ├── documents/           # 업로드된 문서
│   ├── vector_store/        # FAISS 인덱스
│   └── database/            # SQLite DB
│
├── docs/
│   ├── DESIGN.md            # 설계 문서
│   ├── ARCHITECTURE.md      # 아키텍처 문서
│   └── IMPLEMENTATION.md    # 구현 가이드
│
├── .env.example             # 환경 변수 템플릿
├── requirements.txt         # 의존성
└── README.md                # 프로젝트 소개 (이 파일)
```

자세한 아키텍처는 [ARCHITECTURE.md](docs/ARCHITECTURE.md) 참고

---

## 🚀 빠른 시작

### 1. 사전 준비
- Python 3.10 이상
- OpenAI API Key ([발급 방법](https://platform.openai.com/api-keys))

### 2. 설치
```bash
# 1. 레포지토리 클론
git clone https://github.com/your-username/devmate.git
cd devmate

# 2. 가상환경 생성 & 활성화
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일 편집: OPENAI_API_KEY=your-key-here

# 5. DB 초기화
python scripts/init_db.py
```

### 3. 실행
```bash
# 방법 1: Streamlit UI (권장)
streamlit run app/streamlit_app.py

# 방법 2: FastAPI 서버
uvicorn app.main:app --reload
# API 문서: http://localhost:8000/docs
```

### 4. 사용 방법
1. 브라우저에서 `http://localhost:8501` 접속
2. PDF 문서 업로드 (예: FastAPI 공식 문서)
3. 인덱싱 완료 대기 (약 30초)
4. 질문 입력: "FastAPI에서 JWT 인증 어떻게 해?"
5. 답변 + 출처 확인

---

## 📊 성능 지표

### 목표
- **응답 시간**: < 3초 (95th percentile)
- **정확도**: 사용자 만족도 80%+ (5점 척도 4점 이상)
- **비용**: 질문당 < $0.01

### 현재 (Phase 1)
- **평균 응답 시간**: ~2.5초
- **토큰 비용**: 질문당 평균 $0.002
- **인덱싱 속도**: 10페이지 PDF → 30초

---

## 🧪 테스트

```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=app --cov-report=html

# 특정 테스트만
pytest tests/unit/test_models.py -v

# E2E 테스트
pytest tests/e2e/ -v -s
```

---

## 📖 문서

- **[DESIGN.md](docs/DESIGN.md)**: 시스템 설계 및 의사결정
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: 아키텍처 상세
- **[IMPLEMENTATION.md](docs/IMPLEMENTATION.md)**: 구현 가이드
- **[API 문서](http://localhost:8000/docs)**: FastAPI 자동 생성 (서버 실행 필요)

---

## 🗺️ 로드맵

### Phase 1 (완료)
- ✅ 기본 RAG 파이프라인
- ✅ 단일 문서 검색
- ✅ 출처 표시

### Phase 2 (진행 중)
- 🔄 대화 맥락 유지
- 🔄 Streamlit UI
- 🔄 멀티 문서 검색

### Phase 3 (예정)
- 한영 병행 답변
- 코드 하이라이팅
- 성능 최적화

### 향후 계획
- [ ] 다국어 지원 (일본어, 중국어)
- [ ] 음성 입력/출력
- [ ] 이미지 포함 PDF 처리
- [ ] Pinecone 통합 (클라우드 벡터 DB)
- [ ] 사용자 인증/권한 관리

---

## 🤝 기여

이슈 제보, 기능 제안, PR 환영합니다!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

MIT License - 자유롭게 사용하세요!

---

## 👤 작성자

**당신의 이름**
- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com
- LinkedIn: [당신의 이름](https://linkedin.com/in/your-profile)

---

## 🙏 감사의 말

- [LangChain](https://www.langchain.com/) - LLM 애플리케이션 프레임워크
- [FastAPI](https://fastapi.tiangolo.com/) - 현대적인 Python 웹 프레임워크
- [Streamlit](https://streamlit.io/) - 빠른 데이터 앱 구축
- [FAISS](https://github.com/facebookresearch/faiss) - 벡터 유사도 검색

---

## 📞 문의

질문이나 제안이 있으시면 [이슈](https://github.com/your-username/devmate/issues)를 생성해주세요!

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**
