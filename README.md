# HackyMocchi — Autonomous Pentest Agent

> **AI 기반 자율 웹 침투 테스트 에이전트**
> RAG + Hybrid Search + LLM을 결합한 5단계 자동화 파이프라인

---

## Overview

HackyMocchi(해키모찌)는 대상 웹 애플리케이션에 URL 하나만 입력하면 **정찰 → 취약점 검색 → 페이로드 생성 → 공격 실행 → 보고서 출력**까지 5단계를 자율적으로 수행하는 AI 침투 테스트 에이전트입니다.

NVD, Nuclei, PayloadsAllTheThings, ExploitDB 등 대규모 보안 데이터를 벡터 DB에 적재하고, **Hybrid RAG 검색**과 **Groq LLM**을 결합해 상황에 맞는 공격 페이로드를 실시간으로 생성합니다. 모든 과정은 웹 UI에 SSE(Server-Sent Events)로 실시간 스트리밍됩니다.

> **면책 조항**: 이 도구는 **허가된 시스템** 또는 OWASP Juice Shop 등 **교육용 취약 환경**에서만 사용하십시오. 무단 침투 시도는 불법입니다.

---

## Team Members

| 정승윤 | 문선영 | 신대환 | 이정재 |
|:---:|:---:|:---:|:---:|
| AI | AI | AI | AI |

| 이름 | 역할 |
|---|---|
| 정승윤 | 데이터 전처리, LangGraph & LangChain 구현, RAG 성능 향상 |
| 문선영 | 데이터 전처리, Vector DB 적재, Hybrid Search 구현 |
| 신대환 | 데이터 전처리, Vector DB 적재, Reranking 구현 |
| 이정재 | 데이터 전처리, LangGraph & LangChain 구현, UI 제작 |

---

## Architecture

```
[Browser UI]
     │  POST /api/analyze
     ▼
[FastAPI + SSE Stream] ──▶ [Real-time Progress UI]
     │
     ▼
┌──────────────────────────────────────────────────┐
│                 5-Step Pipeline                  │
│                                                  │
│  Step 1  Recon      IP / Server / Tech Stack     │
│  Step 2  Retrieve   Hybrid RAG Search            │
│            ├─ BM25 Keyword Search     (60%)      │
│            ├─ ChromaDB Vector Search  (40%)      │
│            └─ Flashrank Reranker                 │
│  Step 3  Generate   Groq LLM Payload Generation  │
│            └─ Rule-based Fallback                │
│  Step 4  Exploit    HTTP Attack + Retry Loop     │
│            └─ 최대 5회 자율 재시도               │
│  Step 5  Report     Vulnerability Report         │
└──────────────────────────────────────────────────┘
     │
     ▼
[ChromaDB Vector DB]
  ├─ Collection: vuln_knowledge    (취약점 지식)
  └─ Collection: hacking_payloads  (공격 페이로드)
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Backend** | FastAPI, Uvicorn |
| **LLM** | Groq API (`llama-3.1-8b-instant`), Ollama (`llama3.1`) |
| **Orchestration** | LangGraph, LangChain |
| **RAG** | ChromaDB (Vector), BM25Okapi (Keyword), EnsembleRetriever |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Reranker** | Flashrank (`ms-marco-TinyBERT-L-2-v2`) |
| **Frontend** | Vanilla HTML/CSS/JS (Dark / Light 테마) |
| **Streaming** | Server-Sent Events (SSE) |

---

## Vulnerability Data Sources

| 소스 | 설명 |
|---|---|
| [NVD](https://nvd.nist.gov/vuln/data-feeds) | CVE, CVSS, 공식 보안 권고 |
| [Nuclei Templates](https://github.com/projectdiscovery/nuclei) | 템플릿 기반 취약점 탐지 패턴 |
| [PayloadsAllTheThings](https://github.com/swisskyrepo/PayloadsAllTheThings) | SQLi, XSS, RCE 등 공격 페이로드 모음 |
| [Exploit Database](https://gitlab.com/exploit-database/exploitdb) | 공개 익스플로잇 코드 및 공격 사례 |

각 데이터는 전처리 후 임베딩되어 ChromaDB에 적재됩니다.

---

## Quick Start

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Groq API 키 발급: [https://console.groq.com](https://console.groq.com)

`.env`는 `.gitignore`에 등록되어 있어 Git에 업로드되지 않습니다.

### 3. Vector DB 초기화 (최초 1회)

```bash
python load_data.py
```

### 4. 서버 실행

```bash
# Hybrid Search 버전 (권장)
python combination.py

# 기본 버전
python api.py
```

브라우저에서 `http://localhost:8000` 접속

---

## Usage

1. 브라우저에서 `http://localhost:8000` 접속
2. **Target URL** 입력란에 분석할 URL 입력
   ```
   예) http://localhost:3000   ← OWASP Juice Shop
   ```
3. **Analyze** 버튼 클릭
4. 5단계 파이프라인 실시간 진행 상황 확인
5. 완료 후 취약점 분석 보고서 확인

---

## Pipeline Details

### Step 1 — Recon (정찰)
대상 URL의 IP 주소, 서버 소프트웨어(Apache/Nginx/IIS 등), 기술 스택을 자동으로 식별합니다.

### Step 2 — Retrieve (하이브리드 검색)
ChromaDB에 저장된 취약점 지식·공격 페이로드 DB를 검색합니다.
- **BM25 키워드 검색 (60%)**: 정확한 공격 구문 매칭에 강점
- **Vector 유사도 검색 (40%)**: 의미 기반 유사 취약점 탐색
- **Flashrank 리랭킹**: 최종 상위 결과 정밀 필터링
- Groq LLM이 검색어를 기술 스택에 맞게 자동 최적화

### Step 3 — Generate (페이로드 생성)
RAG 결과와 정찰 데이터를 컨텍스트로 Groq LLM에게 HTTP 공격 페이로드(URL, Method, Data)를 생성하도록 요청합니다. LLM 타임아웃/실패 시 규칙 기반 SQLi·XSS 페이로드로 자동 전환합니다.

### Step 4 — Exploit (공격 실행)
생성된 페이로드로 실제 HTTP 요청을 전송하고 응답을 분석합니다. 성공 지표 미탐지 시 Step 3로 되돌아가 최대 **5회** 자율 재시도합니다.

**지원 공격 유형**: SQL Injection, XSS, LFI, IDOR, Command Injection, Auth Bypass

### Step 5 — Report (보고서)
취약점 발견 여부, 사용된 페이로드, 성공 지표, 권장 조치사항을 포함한 보고서를 생성합니다.

---

## Project Structure

```
5th-hackymocchi-ai/
├── api.py              # FastAPI 서버 (기본 버전)
├── combination.py      # FastAPI 서버 (Hybrid Search 버전 — 권장)
├── main.py             # LangGraph 기반 CLI 에이전트
├── hybrid_search.py    # BM25 + ChromaDB 하이브리드 검색 엔진
├── search_engine.py    # 검색 엔진 v1
├── search_engine2.py   # 검색 엔진 v2 (Hybrid + Rerank)
├── load_data.py        # ChromaDB 데이터 로드 스크립트
├── lang_ljj.py         # LangChain 유틸리티
├── multi_query.py      # 멀티쿼리 검색
├── index.html          # 웹 UI (Dark / Light 테마)
├── requirements.txt    # Python 의존성
├── .env                # API 키 (Git 제외)
└── hackymocchi/
    └── chroma_data/    # ChromaDB 벡터 데이터 (Git 제외)
```

---

## Environment Variables

| 변수명 | 설명 | 필수 |
|---|---|:---:|
| `GROQ_API_KEY` | Groq API 인증 키 | ✅ |

---

## Recommended Test Environment

```bash
# OWASP Juice Shop (Docker)
docker run -d -p 3000:3000 bkimminich/juice-shop
```

[OWASP Juice Shop](https://github.com/juice-shop/juice-shop) — Node.js 기반 교육용 취약 웹 앱으로, 이 에이전트의 주요 테스트 대상입니다.

---

## Responsible Use & Legal Notice

### 사용 제한

다음 행위는 엄격히 금지됩니다:

- 허가 없이 타인의 시스템을 스캔하거나 공격하는 행위
- 무단 침투, 서비스 방해(DoS), 데이터 탈취 시도
- 관련 법률을 위반하는 모든 행위
- 상업적 악용 또는 악의적 사용

본 프로젝트는 **명시적으로 허가된 테스트 환경**에서만 사용해야 합니다.

### 권장 사용 환경

- 로컬 테스트 서버 / CTF 환경
- 자체 소유 인프라
- 명시적으로 허가된 보안 테스트 환경 (버그 바운티 등)

### Responsible Disclosure

취약점 발견 시 서비스 제공자에게 비공개로 먼저 통보하고, 패치 기간을 충분히 제공한 뒤 공개하는 것을 권장합니다.

개발자는 사용자에 의해 발생한 법적 문제 및 무단 테스트로 인한 손해에 대해 책임을 지지 않습니다.
