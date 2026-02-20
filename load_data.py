
import os
import sys


os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# ==========================================
# 이제 라이브러리 로드
# ==========================================
import chromadb
from chromadb.utils import embedding_functions
import ijson
import time
import gc
from decimal import Decimal

# ==========================================
# 1. 경로 및 파일 설정 (자동 인식)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = "hackymocchi"

# [1] 데이터 위치
DATA_DIR = os.path.join(BASE_DIR, PROJECT_FOLDER, "data")

# [2] DB 저장 위치
DB_PATH = os.path.join(BASE_DIR, PROJECT_FOLDER, "chroma_data")

# 배치 사이즈
BATCH_SIZE = 1000

# 컬렉션 이름
COLLECTION_KNOWLEDGE = "vuln_knowledge"
COLLECTION_PAYLOADS = "hacking_payloads"

# 파일 목록
FILE_CVE = "hackymocchi_vector_data_optimized.json"
EXPLOIT_FILES = [
    "combined_exploits_final_filtered.json",
    "PayloadAllTheThings_dataset_final.json",
    "nuclei_agent_dataset.json"
]

# ==========================================
# 2. ChromaDB 연결
# ==========================================
print(f" 현재 위치: {BASE_DIR}")
print(f" 타겟 프로젝트: {PROJECT_FOLDER}")
print(f" 데이터 읽는 곳: {DATA_DIR}")
print(f" DB 만드는 곳: {DB_PATH}")

if not os.path.exists(os.path.join(BASE_DIR, PROJECT_FOLDER)):
    print(f"\n 에러: '{PROJECT_FOLDER}' 폴더를 찾을 수 없습니다.")
    print(f"   load_data.py가 '{PROJECT_FOLDER}' 폴더 바로 옆(상위)에 있는지 확인해주세요.")
    exit()

try:
    # 기존 코드에서 path 설정 유지
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=chromadb.config.Settings(
            is_persistent=True,
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # 임베딩 모델 로드
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu" # 맥북 충돌 방지용 명시
    )
    
    coll_know = client.get_or_create_collection(name=COLLECTION_KNOWLEDGE, embedding_function=emb_fn)
    coll_pay = client.get_or_create_collection(name=COLLECTION_PAYLOADS, embedding_function=emb_fn)
    
    print(f" DB 연결 성공!")
    
except Exception as e:
    print(f" DB 초기화 실패: {e}")
    print(" 팁: 만약 pysqlite3 에러라면 'pip install pysqlite3-binary'를 시도해보세요.")
    exit()

# ==========================================
# 3. 유틸리티 함수
# ==========================================
def convert_decimal(value):
    if isinstance(value, Decimal): return float(value)
    return value

def count_total_items(full_path):
    if not os.path.exists(full_path): return 0
    print(f" 스캔 중: {os.path.basename(full_path)} ...", end="", flush=True)
    count = 0
    try:
        with open(full_path, 'rb') as f:
            for _ in ijson.items(f, 'item'): count += 1
        print(f" -> {count:,} 건")
        return count
    except:
        print(" -> (실패)")
        return 0

# ==========================================
# 4. 적재 엔진
# ==========================================
def ingest_stream_data(filename, data_type):
    full_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(full_path):
        print(f" 파일 없음: {full_path}")
        return

    target_collection = coll_know if data_type == 'cve' else coll_pay
    
    existing_count = target_collection.count()

    total_count = count_total_items(full_path)
    if total_count == 0: return

    file_prefix = os.path.basename(full_path).split('.')[0]
    skip_count = 0
    if data_type == 'cve' and existing_count > 0:
        print(f" [이어하기 감지] 이미 {existing_count:,}개가 DB에 있습니다.")
        print(f" 앞부분 {existing_count:,}건을 건너뛰고 시작합니다...")
        skip_count = existing_count

    print(f" [{file_prefix}] 적재 시작 (타겟: {target_collection.name})")
    
    ids, docs, metas = [], [], []
    processed_count = 0
    saved_count = 0 
    start_time = time.time()

    try:
        with open(full_path, 'rb') as f:
            for item in ijson.items(f, 'item'):
                processed_count += 1

                if data_type == 'cve' and processed_count <= skip_count:
                    if processed_count % 10000 == 0:
                        print(f"    {processed_count:,} 건 스킵 중...")
                    continue

                # [A] CVE
                if data_type == 'cve':
                    doc_id = f"cve_{item.get('id')}"
                    doc_text = item.get('text', '')
                    raw_meta = item.get('metadata', {})
                    clean_meta = {k: convert_decimal(v) for k, v in raw_meta.items()}
                    clean_meta['source'] = file_prefix

                # [B] Exploit
                else:
                    doc_id = f"exploit_{file_prefix}_{processed_count}"
                    raw_payload = item.get('payload', item.get('payloads', ''))
                    payloads_str = "\n".join(raw_payload) if isinstance(raw_payload, list) else str(raw_payload)
                    desc = item.get('description', 'No description')
                    doc_text = f"Attack Description: {desc}\nPayload Code: {payloads_str}"
                    clean_meta = {
                        "source": file_prefix,
                        "category": str(item.get('category', 'Unknown')),
                        "method": str(item.get('method', 'Unknown'))
                    }

                ids.append(doc_id)
                docs.append(doc_text)
                metas.append(clean_meta)

                if len(ids) >= BATCH_SIZE:
                    try:
                        target_collection.add(ids=ids, documents=docs, metadatas=metas)
                        saved_count += len(ids)
                    except Exception as e:
                        pass
                        
                    ids, docs, metas = [], [], []
                    gc.collect() 
                    
                    if saved_count % 1000 == 0: # 로그 너무 자주 찍히지 않게 100 -> 1000으로 조정
                         print(f"    {saved_count:,}개 신규 저장 (진행: {processed_count:,}/{total_count:,})")

            if ids:
                try:
                    target_collection.add(ids=ids, documents=docs, metadatas=metas)
                except: pass
                gc.collect()

        duration = time.time() - start_time
        print(f" [{file_prefix}] 완료! ({duration:.2f}초)")

    except Exception as e:
        print(f" [{file_prefix}] 에러: {e}")

# ==========================================
# 5. 실행
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("      Hackymocchi 통합 데이터 적재기 (Fix Ver.)")
    print("="*50)
    
    # 1. 지식 데이터 적재
    ingest_stream_data(FILE_CVE, 'cve')
    
    print("-" * 30)
    
    # 2. 무기 데이터 적재
    for f in EXPLOIT_FILES:
        ingest_stream_data(f, 'exploit')
    
    print("\n" + "="*50)
    try:
        print(f" 지식 DB 저장량: {coll_know.count():,} 개")
        print(f" 무기 DB 저장량: {coll_pay.count():,} 개")
    except:
        pass
    print(f" 최종 저장 위치: {DB_PATH}")
    print("="*50)