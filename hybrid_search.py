import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import re
import time
from datetime import datetime

# --- [System Logger Configuration] ---
def sys_log(level, module, msg):
    # 포맷: [YYYY-MM-DD HH:MM:SS] [LEVEL] [MODULE] Message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level:<5}] [{module:<10}] {msg}")

# --- [Engine Initialization] ---
print("\n" + "="*110)
sys_log("INFO", "KERNEL", "Initializing Dual-Stream Retrieval System...")

db_path = "./hackymocchi/chroma_data"
client = chromadb.PersistentClient(path=db_path)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="vuln_knowledge", embedding_function=emb_fn)

sys_log("INFO", "STORAGE", f"Attached to Vector Store: {db_path}")

# BM25 Index Build (SQLite 변수 한도(999개) 초과 방지를 위해 배치 로드)
sys_log("INFO", "INDEXER", "Building In-Memory Sparse Index...")
start_t = time.time()
documents = []
ids = []
_BATCH = 500
_offset = 0
while True:
    batch = collection.get(limit=_BATCH, offset=_offset)
    if not batch['documents']:
        break
    documents.extend(batch['documents'])
    ids.extend(batch['ids'])
    _offset += _BATCH

def tokenize(text):
    return re.findall(r'\w+', text.lower())

tokenized_corpus = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
sys_log("INFO", "INDEXER", f"Indexing Complete. Docs: {len(documents)}. Time: {time.time()-start_t:.4f}s")
print("="*110 + "\n")


# --- [Pattern Matcher] ---
def check_pattern_match(query, content):
    # 쿼리의 핵심 식별자(ID)가 본문에 포함되어 있는지 단순 검사
    # (Re-ranking이 아니라 단순 텍스트 매칭 여부 확인용)
    identifier = query.split()[0] # 첫 어절 (예: CVE-xxxx)
    return identifier.lower() in content.lower()

# --- [Pure Hybrid Retrieval Function] ---
def execute_hybrid_retrieval(query_text, limit=3):
    print(f"┌{'─'*108}┐")
    sys_log("QUERY", "INPUT", f"Payload: '{query_text}'")
    
    # --- [Stream A] Semantic Vector Retrieval ---
    sys_log("INFO", "STREAM_A", "Fetching from Dense Vector Space...")
    t0 = time.time()
    vec_results = collection.query(query_texts=[query_text], n_results=limit)
    t_vec = time.time() - t0
    
    v_docs = vec_results['documents'][0]
    v_ids = vec_results['ids'][0]
    v_dists = vec_results['distances'][0]

    # --- [Stream B] Lexical BM25 Retrieval ---
    sys_log("INFO", "STREAM_B", "Fetching from Sparse Keyword Index...")
    t0 = time.time()
    query_tokens = tokenize(query_text)
    # 점수 계산 없이 상위 N개만 빠르게 추출
    bm25_top = bm25.get_top_n(query_tokens, documents, n=limit)
    # BM25 결과의 ID를 찾기 위한 역매핑 (성능 최적화 필요하나 데모용으로 순차 검색)
    # 실제 운영 환경에서는 인덱스 매핑을 미리 해둠
    b_indices = [documents.index(doc) for doc in bm25_top]
    b_ids = [ids[i] for i in b_indices]
    t_bm25 = time.time() - t0

    # --- [System Output] ---
    # Ranking이나 Winner를 가리지 않고, 두 스트림의 결과를 있는 그대로 출력
    
    print("-" * 110)
    print(f" >> [STREAM A: Vector Engine] Latency: {t_vec*1000:.2f}ms")
    print(f"    {'IDX':<4} | {'MATCH':<7} | {'ID':<15} | {'CONTENT SNAPSHOT'}")
    
    for i in range(len(v_docs)):
        match_flag = "[YES]" if check_pattern_match(query_text, v_docs[i]) else "[NO ]"
        # Ranking 점수가 아니라 Distance(거리) 정보만 표기
        print(f"    {i+1:<4} | {match_flag:<7} | {v_ids[i]:<15} | {v_docs[i][:60]}...")

    print("-" * 110)
    print(f" >> [STREAM B: BM25 Engine]   Latency: {t_bm25*1000:.2f}ms")
    print(f"    {'IDX':<4} | {'MATCH':<7} | {'ID':<15} | {'CONTENT SNAPSHOT'}")
    
    for i in range(len(bm25_top)):
        match_flag = "[YES]" if check_pattern_match(query_text, bm25_top[i]) else "[NO ]"
        print(f"    {i+1:<4} | {match_flag:<7} | {b_ids[i]:<15} | {bm25_top[i][:60]}...")

    print("-" * 110)
    
    # --- [Coverage Analysis] ---
    # 순위가 아니라 '교집합(Coverage)'만 분석
    set_a = set(v_ids)
    set_b = set(b_ids)
    common = set_a.intersection(set_b)
    
    sys_log("DATA", "COVERAGE", f"Unique Docs Found: {len(set_a.union(set_b))}")
    sys_log("DATA", "OVERLAP", f"Common Docs: {len(common)} {list(common) if common else ''}")
    
    print(f"└{'─'*108}┘\n")

# --- [Execution] ---
if __name__ == "__main__":
    # Test 1: Vector가 놓치고 BM25가 잡는 경우
    execute_hybrid_retrieval("CVE-2022-22293")
    
    # Test 2: 둘 다 잡거나 Vector가 유리한 경우
    execute_hybrid_retrieval("SQL Injection bypass payload")


# --- [Main.py 전용 데이터 배달 함수] ---
def get_documents_for_llm(query_text, top_k=5):
    # 1. Vector 검색
    vector_results = collection.query(query_texts=[query_text], n_results=top_k)
    v_docs = vector_results['documents'][0]

    # 2. BM25 검색
    query_tokens = tokenize(query_text)
    bm25_top = bm25.get_top_n(query_tokens, documents, n=top_k)

    # 3. 중복 제거 후 리스트로 반환
    final_docs = list(set(v_docs + bm25_top))
    return final_docs
