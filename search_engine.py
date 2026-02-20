import os
import warnings
from groq import Groq
warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

class SecuritySearchEngine:
    def __init__(self, db_path="./chroma_data"):
        print("[*] 초고속 엔진 초기화 중 (Groq API + Llama 3.1 8B)...")
        
        # 1. 벡터 DB 및 리랭커 설정 (기존 데이터 완벽 연동)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vs_payloads = Chroma(collection_name="hacking_payloads", persist_directory=db_path, embedding_function=self.embeddings)
        self.vs_vuln = Chroma(collection_name="vuln_knowledge", persist_directory=db_path, embedding_function=self.embeddings)
        self.compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
        
        # 2. Groq 클라이언트 설정 (LangChain 래퍼 없이 직접 연결)
        # 발급받은 API 키를 여기에 입력하세요.
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Llama 3.1 8B의 Groq 버전 모델명입니다.
        self.model_name = "llama-3.1-8b-instant" 
        print("[+] 준비 완료! 이제 1~2초 만에 완벽한 키워드를 뽑아냅니다.")

    def get_best_docs(self, user_query: str, k_candidates: int = 15, top_n: int = 3):
# 3. 프롬프트 다이어트 (핵심 검색어 3~4개로 강제 제한)
        prompt = (
            f"Translate the following Korean query into a concise English search phrase (maximum 4 words) for a CVE database.\n"
            f"Query: {user_query}\n"
            f"CRITICAL RULE 1: Output ONLY English words. No Korean characters.\n"
            f"CRITICAL RULE 2: Just output 3-4 words separated by spaces. NO commas, NO redundant synonyms.\n"
            f"Search Phrase:"
        )
        
        try:
            # Groq API 호출 (초고속 추론)
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1, # 일관성 있는 키워드 추출을 위해 낮게 설정
                max_tokens=50    # 속도를 위해 최대 출력 길이 제한
            )
            eng_query = response.choices[0].message.content.strip()
            
            # 만약 모델이 말을 안 듣고 줄바꿈을 썼을 경우를 대비한 안전장치
            eng_query = eng_query.replace('\n', ', ').replace('**', '')
            
        except Exception as e:
            print(f"[-] API 호출 에러: {e}")
            eng_query = user_query
            
        print(f"\n[*] 입력 질문: {user_query}")
        print(f"[*] 추출 키워드: {eng_query}") 
        
        # 4. 검색 및 리랭킹
        docs_payloads = self.vs_payloads.similarity_search(eng_query, k=k_candidates)
        docs_vuln = self.vs_vuln.similarity_search(eng_query, k=k_candidates)
        base_docs = docs_payloads + docs_vuln
        
        if not base_docs: return []
        
        # 리랭커를 통해 최종 정확도 필터링
        return self.compressor.compress_documents(documents=base_docs, query=eng_query)[:top_n]

if __name__ == "__main__":
    engine = SecuritySearchEngine()
    while True:
        query = input("\n[?] 질문 입력 (q: 종료): ")
        if query.lower() == 'q': break
        
        results = engine.get_best_docs(query)
        
        print("\n" + "━"*70)
        print(f" 🏆 최적의 보안 문서 검색 결과 (Top {len(results)})")
        print("━"*70)
        
        if not results:
            print("  [!] 일치하는 보안 문서가 없습니다.")
        else:
            for i, doc in enumerate(results):
                score = doc.metadata.get("relevance_score", 0.0)
                
                # 1. 원본 내용을 그대로 가져옵니다.
                content = doc.page_content
                
                # 2. 보기 편하도록 주요 항목 앞에 줄바꿈(\n)과 기호를 넣어줍니다.
                content = content.replace("Severity:", "\n    🔸 Severity:")
                content = content.replace("Attack Vector:", "\n    🔸 Attack Vector:")
                content = content.replace("Privileges Required:", "\n    🔸 Privileges Required:")
                content = content.replace("Description:", "\n    📝 Description:")
                content = content.replace("Payload Code:", "\n    💻 Payload Code:")
                
                print(f" [{i+1}위] 🎯 매칭 점수: {score:.3f}")
                print(f" 📄 추출된 내용: {content[:350]}...") # 길이를 조금 늘려서 충분히 보이게 합니다.
                print("─" * 70)