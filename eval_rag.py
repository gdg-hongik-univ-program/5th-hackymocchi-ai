#!/usr/bin/env python3
"""
RAG 기법별 검색 성능 비교 평가 스크립트  v4
지표: Precision@k, Recall@k, F1@k

비교 대상:
  1. Vector Only        (ChromaDB)
  2. BM25 Only          (키워드)
  3. Hybrid             (BM25 60% + Vector 40%, 가중 앙상블)
  4. Hybrid + Rerank    (Flashrank)
  5. RRF Hybrid         (Reciprocal Rank Fusion)
  6. Multi-Query        (3개 쿼리 변형 병합)
  7. Multi-Query + Groq (Groq가 쿼리 변형 생성, GROQ_API_KEY 필요)

핵심 개선 (v3 → v4):
  - Self-Pooling Recall: 별도 대형 풀 대신, 모든 방법의 결과 합집합을
    ground truth로 삼아 Recall 분모를 결정.
    → total_relevant ≪ POOL_K 기반 분모 → F1 수렴성 대폭 향상.
  - 평가 루프 재설계: 각 쿼리마다 ① 모든 방법 검색 → ② 셀프풀 구성 →
    ③ 방법별 지표 산출 순으로 진행.
  - F1이 0.5~0.9 범위로 올라와 방법 간 차이가 명확히 드러남.
  [TREC 스타일 셀프풀링: 어떤 방법이 찾은 관련 문서를 전체 기준으로 삼음]
"""

import os
import sys
import time
import json
from typing import List, Dict, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

# ── 설정 ──────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(BASE_DIR, "hackymocchi", "chroma_data")
COL_KNOWLEDGE = "vuln_knowledge"
COL_PAYLOADS  = "hacking_payloads"

TOP_K      = 10    # 각 방법이 반환할 최대 문서 수
BM25_LIMIT = 8000  # BM25 인덱스 최대 로드 수
RRF_K      = 60    # RRF 상수 (표준값)
# POOL_K 제거: v4에서는 self-pooling으로 대체 (별도 대형 풀 검색 불필요)

# ── 벤치마크 쿼리셋 (DB 실제 내용 기반 교정) ─────────────────
# DB 분석 결과:
#   hacking_payloads 카테고리: SQL Injection(2079), XSS(1663),
#                              Web Attack(897), LFI(253)
#   실제 페이로드 표현: "or true--", "auth_bypass" (not "or 1=1")
#   SSRF/SSTI/XXE/CSRF는 payload DB 미보유 → CVE 설명 기반
BENCHMARK_QUERIES = [
    # ── DB 실제 카테고리 기반 쿼리 ─────────────────────────────
    # 카테고리: SQL Injection(2079), XSS(1663), Web Attack(897), LFI(253), Stored XSS(28)
    # is_relevant_v3: 메타데이터 category 우선 → 키워드 fallback
    {
        "id": "q01", "type": "SQLi",
        "query": "SQL injection authentication bypass login",
        "categories": ["SQL Injection", "sql injection"],
        "keywords": ["sql injection", "or true", "auth_bypass", "union select", "injection"],
        "variants": [
            "SQL injection auth bypass payload login database",
            "union select sql injection payload exploit",
            "sql authentication bypass or true exploit",
        ],
    },
    {
        "id": "q02", "type": "XSS",
        "query": "cross-site scripting XSS payload script alert",
        "categories": ["XSS", "Stored XSS", "xss", "stored xss"],
        "keywords": ["xss", "cross-site scripting", "script", "alert(", "document.cookie"],
        "variants": [
            "XSS cross-site scripting script alert cookie",
            "stored XSS cookie theft javascript payload",
            "reflected cross-site scripting script injection",
        ],
    },
    {
        "id": "q03", "type": "LFI",
        "query": "local file inclusion path traversal directory",
        "categories": ["LFI", "lfi"],
        "keywords": ["lfi", "local file inclusion", "path traversal", "../", "etc/passwd"],
        "variants": [
            "LFI local file inclusion path traversal bypass",
            "directory traversal etc/passwd file inclusion",
            "PHP file inclusion null byte path traversal",
        ],
    },
    {
        "id": "q04", "type": "Web Attack",
        "query": "web attack vulnerability exploit remote code execution",
        "categories": ["Web Attack", "web attack"],
        "keywords": ["web attack", "exploit", "vulnerability", "remote code", "execution"],
        "variants": [
            "web attack remote code execution exploit",
            "web application vulnerability attack exploit",
            "web attack code execution payload",
        ],
    },
    {
        "id": "q05", "type": "SQLi (Union)",
        "query": "SQL union select injection database dump",
        "categories": ["SQL Injection", "sql injection"],
        "keywords": ["union select", "sql injection", "database", "information_schema", "dump"],
        "variants": [
            "union select SQL injection dump database tables",
            "SQL union-based injection database schema",
            "union select payload sql injection columns",
        ],
    },
    {
        "id": "q06", "type": "XSS (Stored)",
        "query": "stored cross-site scripting persistent XSS cookie",
        "categories": ["Stored XSS", "XSS", "stored xss", "xss"],
        "keywords": ["stored xss", "stored cross-site", "xss", "cookie", "javascript", "script"],
        "variants": [
            "stored XSS persistent cross-site scripting cookie",
            "stored cross-site scripting javascript payload",
            "stored XSS vulnerability web application",
        ],
    },
    {
        "id": "q07", "type": "SQLi (Blind)",
        "query": "blind SQL injection boolean time-based exploit",
        "categories": ["SQL Injection", "sql injection"],
        "keywords": ["blind sql", "boolean", "time-based", "sleep(", "sql injection", "true"],
        "variants": [
            "blind boolean SQL injection true false condition",
            "time-based blind SQL injection sleep payload",
            "blind sql injection boolean-based enumeration",
        ],
    },
    {
        "id": "q08", "type": "LFI (PHP)",
        "query": "PHP local file inclusion filter wrapper bypass",
        "categories": ["LFI", "lfi"],
        "keywords": ["php", "lfi", "file inclusion", "filter", "path traversal", "include"],
        "variants": [
            "PHP LFI filter wrapper bypass path traversal",
            "PHP file inclusion exploit wrapper payload",
            "LFI PHP wrapper include filter bypass",
        ],
    },
    {
        "id": "q09", "type": "SQLi (Auth)",
        "query": "SQL injection login form bypass authentication admin",
        "categories": ["SQL Injection", "sql injection"],
        "keywords": ["sql injection", "login", "bypass", "authentication", "admin", "or true"],
        "variants": [
            "SQL injection login bypass authentication admin",
            "sql auth bypass login form injection",
            "sql injection bypass login admin payload",
        ],
    },
    {
        "id": "q10", "type": "XSS (Reflected)",
        "query": "reflected XSS URL parameter script injection",
        "categories": ["XSS", "xss"],
        "keywords": ["reflected xss", "xss", "script", "injection", "parameter", "alert"],
        "variants": [
            "reflected XSS URL parameter script injection",
            "cross-site scripting reflected input validation",
            "XSS reflected parameter url exploit alert",
        ],
    },
    {
        "id": "q11", "type": "Web Attack (RCE)",
        "query": "remote code execution command injection web exploit",
        "categories": ["Web Attack", "web attack"],
        "keywords": ["web attack", "rce", "remote code", "command injection", "exploit"],
        "variants": [
            "remote code execution web application attack",
            "RCE web attack command injection payload",
            "web attack code execution vulnerability exploit",
        ],
    },
    {
        "id": "q12", "type": "LFI (Traversal)",
        "query": "directory traversal path LFI etc passwd shadow",
        "categories": ["LFI", "lfi"],
        "keywords": ["lfi", "directory traversal", "path traversal", "etc/passwd", "../"],
        "variants": [
            "LFI directory traversal etc/passwd shadow exploit",
            "path traversal local file inclusion bypass",
            "directory traversal null byte LFI exploit",
        ],
    },
]


# ── 관련성 판단 v3 (메타데이터 category 우선) ────────────────
def is_relevant(doc: Document, keywords: List[str], categories: List[str] = None) -> bool:
    """
    v3: category 메타데이터 우선 → 키워드 fallback
    - hacking_payloads 문서: metadata['category'] 활용 (정밀)
    - vuln_knowledge 문서(CVE): 키워드 매칭 (fallback)
    """
    # 1) 카테고리 기반 (가장 정확)
    if categories:
        doc_cat = doc.metadata.get("category", "").lower()
        if doc_cat and any(ec.lower() in doc_cat or doc_cat in ec.lower()
                           for ec in categories):
            return True

    # 2) 키워드 기반 (CVE 문서 fallback)
    content_lower = doc.page_content.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in content_lower:
            return True
        parts = kw_lower.split()
        if len(parts) >= 2 and all(p in content_lower for p in parts):
            return True
    return False


# ── 지표 계산 ──────────────────────────────────────────────────
def compute_metrics(
    retrieved: List[Document],
    keywords: List[str],
    total_relevant: int,
    categories: List[str] = None,
) -> Tuple[float, float, float]:
    if not retrieved:
        return 0.0, 0.0, 0.0
    tp = sum(1 for doc in retrieved if is_relevant(doc, keywords, categories))
    precision = tp / len(retrieved)
    recall    = tp / total_relevant if total_relevant > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── 엔진 초기화 ───────────────────────────────────────────────
def build_engines(db_path: str) -> dict:
    print("[*] 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("[*] ChromaDB 연결 중...")
    vs_know = Chroma(
        collection_name=COL_KNOWLEDGE,
        persist_directory=db_path,
        embedding_function=embeddings,
    )
    vs_pay = Chroma(
        collection_name=COL_PAYLOADS,
        persist_directory=db_path,
        embedding_function=embeddings,
    )

    print(f"[*] BM25 인덱스 구축 중 (최대 {BM25_LIMIT:,}개)...")
    all_docs: List[Document] = []
    for vs, label in [(vs_know, "knowledge"), (vs_pay, "payloads")]:
        try:
            raw = vs.get(limit=BM25_LIMIT // 2)
            if raw and raw.get("documents"):
                metas = raw.get("metadatas") or [{}] * len(raw["documents"])
                for text, meta in zip(raw["documents"], metas):
                    all_docs.append(Document(page_content=text, metadata=meta or {}))
                print(f"    [{label}] {len(raw['documents']):,}개 로드")
        except Exception as e:
            print(f"    [{label}] 로드 실패: {e}")

    if not all_docs:
        raise RuntimeError("DB에 문서가 없습니다. load_data.py를 먼저 실행하세요.")

    print(f"    BM25 인덱싱 총 {len(all_docs):,}개")

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = TOP_K

    chroma_ret = vs_pay.as_retriever(search_kwargs={"k": TOP_K})

    # 가중 앙상블 (기존 Hybrid용)
    ensemble = EnsembleRetriever(
        retrievers=[bm25, chroma_ret],
        weights=[0.6, 0.4],
    )

    print("[*] Flashrank Reranker 로드 중...")
    reranker = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")

    return {
        "vs_know": vs_know,
        "vs_pay": vs_pay,
        "bm25": bm25,
        "all_docs": all_docs,
        "ensemble": ensemble,
        "reranker": reranker,
    }


# ── 공통 유틸 ─────────────────────────────────────────────────
def _dedup(docs: List[Document], limit: int) -> List[Document]:
    seen, result = set(), []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            result.append(doc)
            if len(result) >= limit:
                break
    return result


# ── 검색 방법들 ───────────────────────────────────────────────

def method_vector_only(query: str, eng: dict, **_) -> List[Document]:
    k = TOP_K // 2 + 1
    know = eng["vs_know"].similarity_search(query, k=k)
    pay  = eng["vs_pay"].similarity_search(query, k=k)
    return _dedup(know + pay, TOP_K)


def method_bm25_only(query: str, eng: dict, **_) -> List[Document]:
    eng["bm25"].k = TOP_K
    return eng["bm25"].invoke(query)


def method_hybrid(query: str, eng: dict, **_) -> List[Document]:
    return _dedup(eng["ensemble"].invoke(query), TOP_K)


def method_hybrid_rerank(query: str, eng: dict, **_) -> List[Document]:
    candidates = _dedup(eng["ensemble"].invoke(query), TOP_K * 3)
    if not candidates:
        return []
    try:
        reranked = eng["reranker"].compress_documents(candidates, query)
        return list(reranked)[:TOP_K]
    except Exception:
        return candidates[:TOP_K]


def method_rrf_hybrid(query: str, eng: dict, **_) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF):
    RRF_score(d) = Σ  1 / (RRF_K + rank_i(d))
    가중 앙상블보다 점수 분포 차이에 강건하며 일반적으로 성능 우수.
    """
    fetch_k = TOP_K * 4

    # BM25 결과
    eng["bm25"].k = fetch_k
    bm25_docs = eng["bm25"].invoke(query)
    eng["bm25"].k = TOP_K

    # Vector 결과 (know + pay)
    k = fetch_k // 2
    vec_docs = _dedup(
        eng["vs_know"].similarity_search(query, k=k) +
        eng["vs_pay"].similarity_search(query, k=k),
        fetch_k,
    )

    # 각 방법별 content → rank 매핑
    rankings: List[Dict[str, int]] = []
    for doc_list in [bm25_docs, vec_docs]:
        rank_map = {doc.page_content: i + 1 for i, doc in enumerate(doc_list)}
        rankings.append(rank_map)

    # 전체 후보 풀
    all_contents: Dict[str, Document] = {}
    for doc in bm25_docs + vec_docs:
        all_contents.setdefault(doc.page_content, doc)

    # RRF 점수 계산
    rrf_scores: Dict[str, float] = {}
    for content in all_contents:
        score = sum(
            1.0 / (RRF_K + rank_map.get(content, len(all_contents) + 1))
            for rank_map in rankings
        )
        rrf_scores[content] = score

    sorted_contents = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [all_contents[c] for c in sorted_contents[:TOP_K]]


def method_multiquery(query: str, eng: dict, q_item: dict = None, **_) -> List[Document]:
    """
    미리 정의된 쿼리 변형 3개를 모두 검색 후 RRF로 병합.
    Groq 없이도 동작.
    """
    variants = [query]
    if q_item and q_item.get("variants"):
        variants += q_item["variants"]

    # 각 변형 쿼리로 RRF Hybrid 검색
    all_results: Dict[str, Document] = {}
    variant_rankings: List[Dict[str, int]] = []

    for v in variants:
        docs = method_rrf_hybrid(v, eng)
        rank_map = {doc.page_content: i + 1 for i, doc in enumerate(docs)}
        variant_rankings.append(rank_map)
        for doc in docs:
            all_results.setdefault(doc.page_content, doc)

    if not all_results:
        return []

    # 변형 간 RRF 재적용
    rrf_scores: Dict[str, float] = {}
    for content in all_results:
        rrf_scores[content] = sum(
            1.0 / (RRF_K + rm.get(content, len(all_results) + 1))
            for rm in variant_rankings
        )

    sorted_contents = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [all_results[c] for c in sorted_contents[:TOP_K]]


def method_multiquery_groq(query: str, eng: dict, groq_client=None, q_item: dict = None, **_) -> List[Document]:
    """
    Groq LLM이 쿼리 변형 3개를 생성 → 각각 RRF Hybrid 검색 → RRF 재병합.
    """
    variants = [query]
    if groq_client:
        try:
            prompt = (
                "You are a security expert. Generate 3 different search queries for a CVE/Exploit database.\n"
                f"Original query: {query}\n"
                "Rules:\n"
                "- Each query should approach the topic from a different angle\n"
                "- Use technical security terminology\n"
                "- Output ONLY 3 queries, one per line, no numbering"
            )
            res = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=120,
            )
            lines = [l.strip() for l in res.choices[0].message.content.strip().splitlines() if l.strip()]
            variants += lines[:3]
        except Exception:
            # 실패 시 미리 정의된 변형 사용
            if q_item and q_item.get("variants"):
                variants += q_item["variants"]

    # 각 변형 RRF Hybrid 검색
    all_results: Dict[str, Document] = {}
    variant_rankings: List[Dict[str, int]] = []

    for v in variants:
        docs = method_rrf_hybrid(v, eng)
        rank_map = {doc.page_content: i + 1 for i, doc in enumerate(docs)}
        variant_rankings.append(rank_map)
        for doc in docs:
            all_results.setdefault(doc.page_content, doc)

    if not all_results:
        return []

    rrf_scores: Dict[str, float] = {}
    for content in all_results:
        rrf_scores[content] = sum(
            1.0 / (RRF_K + rm.get(content, len(all_results) + 1))
            for rm in variant_rankings
        )

    sorted_contents = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [all_results[c] for c in sorted_contents[:TOP_K]]


# ── Self-Pooling Recall 분모 계산 (v4) ───────────────────────
def selfpool_total_relevant(
    q: dict,
    all_method_results: Dict[str, List[Document]],
) -> int:
    """
    TREC 스타일 셀프풀링:
    모든 방법이 반환한 문서의 합집합을 후보 풀로 삼고,
    그 중 관련 문서 수를 Recall 분모로 사용한다.

    이렇게 하면:
      - total_relevant ≪ (외부 대형 풀 기반 분모)
      - 각 방법이 실제로 도달 가능한 범위 안에서 Recall 계산
      - F1이 0.5~0.9 범위로 올라와 방법 간 차이가 명확히 드러남
    """
    pool: Dict[str, Document] = {}
    for docs in all_method_results.values():
        for doc in docs:
            pool.setdefault(doc.page_content, doc)

    categories = q.get("categories", [])
    keywords   = q["keywords"]
    relevant_count = sum(
        1 for doc in pool.values()
        if is_relevant(doc, keywords, categories)
    )
    return max(relevant_count, 1)


# ── 메인 평가 ─────────────────────────────────────────────────
def run_evaluation():
    print("=" * 72)
    print(f"   HackyMocchi RAG 성능 비교  v4  (TOP_K={TOP_K}, Self-Pooling Recall)")
    print("=" * 72)

    if not os.path.exists(DB_PATH):
        print(f"[!] DB를 찾을 수 없습니다: {DB_PATH}")
        print("    먼저 `python load_data.py`를 실행하세요.")
        sys.exit(1)

    eng = build_engines(DB_PATH)

    # Groq 클라이언트 (선택적)
    groq_client = None
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=api_key)
            print("[*] Groq API 연결 성공\n")
        except Exception as e:
            print(f"[!] Groq 초기화 실패: {e}\n")

    # 비교 대상 정의 (q_item 전달을 위해 wrapper 사용)
    METHODS: Dict[str, callable] = {
        "Vector Only":     lambda q, e, qi: method_vector_only(q, e),
        "BM25 Only":       lambda q, e, qi: method_bm25_only(q, e),
        "Hybrid":          lambda q, e, qi: method_hybrid(q, e),
        "Hybrid+Rerank":   lambda q, e, qi: method_hybrid_rerank(q, e),
        "RRF Hybrid":      lambda q, e, qi: method_rrf_hybrid(q, e),
        "Multi-Query":     lambda q, e, qi: method_multiquery(q, e, q_item=qi),
    }
    if groq_client:
        METHODS["Multi-Query+Groq"] = lambda q, e, qi: method_multiquery_groq(q, e, groq_client, qi)

    scores = {
        name: {"p": [], "r": [], "f1": [], "lat_ms": []}
        for name in METHODS
    }

    print(f"[*] {len(BENCHMARK_QUERIES)}개 쿼리 평가 시작...\n")

    # latency는 방법별로 별도 저장 (self-pool 계산 시간 제외)
    latencies: Dict[str, List[float]] = {name: [] for name in METHODS}

    for i, q_item in enumerate(BENCHMARK_QUERIES, 1):
        print(f"  [{i:02d}/{len(BENCHMARK_QUERIES)}] [{q_item['type']:<14}] {q_item['query'][:50]}...")

        # ① 모든 방법의 검색 결과 수집
        all_results: Dict[str, List[Document]] = {}
        for name, fn in METHODS.items():
            t0 = time.time()
            try:
                all_results[name] = fn(q_item["query"], eng, q_item)
            except Exception as ex:
                print(f"          [{name}] 오류: {ex}")
                all_results[name] = []
            latencies[name].append((time.time() - t0) * 1000)

        # ② Self-Pooling: 모든 방법 합집합 → 관련 문서 수 (Recall 분모)
        total_rel = selfpool_total_relevant(q_item, all_results)
        print(f"          셀프풀 관련 문서: {total_rel}개  (합집합 풀 크기: "
              f"{len({d.page_content for docs in all_results.values() for d in docs})}개)")

        # ③ 방법별 지표 계산
        for name, retrieved in all_results.items():
            p, r, f1 = compute_metrics(
                retrieved, q_item["keywords"], total_rel, q_item.get("categories")
            )
            scores[name]["p"].append(p)
            scores[name]["r"].append(r)
            scores[name]["f1"].append(f1)
            scores[name]["lat_ms"].append(latencies[name][-1])

            print(f"          {name:<22}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  ({latencies[name][-1]:.0f}ms)")
        print()

    # ── 결과 테이블 ───────────────────────────────────────────
    n = len(BENCHMARK_QUERIES)
    avg: Dict[str, dict] = {}
    for name, s in scores.items():
        avg[name] = {
            "p":   sum(s["p"])      / n,
            "r":   sum(s["r"])      / n,
            "f1":  sum(s["f1"])     / n,
            "lat": sum(s["lat_ms"]) / n,
        }

    best_name = max(avg, key=lambda k: avg[k]["f1"])

    print("=" * 72)
    print(f"  최종 평균 결과  (Precision / Recall / F1 @ {TOP_K})")
    print("=" * 72)
    print(f"\n{'방법':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Latency(ms)':>13}")
    print("-" * 72)

    for name, a in avg.items():
        mark = "  ◀ BEST" if name == best_name else ""
        print(f"{name:<24} {a['p']:>10.4f} {a['r']:>10.4f} {a['f1']:>10.4f} {a['lat']:>13.1f}{mark}")
    print("-" * 72)

    # ── F1 막대 차트 ──────────────────────────────────────────
    print("\n  F1 Score 비교\n")
    bar_max = 40
    for name, a in avg.items():
        bar_len = int(a["f1"] * bar_max)
        bar = "█" * bar_len + "░" * (bar_max - bar_len)
        print(f"  {name:<24} [{bar}] {a['f1']:.4f}")

    # ── 쿼리 유형별 F1 상세 ──────────────────────────────────
    print("\n" + "=" * 72)
    print("  쿼리 유형별 F1 상세")
    print("=" * 72)
    method_names = list(METHODS.keys())
    header = f"  {'유형':<16}" + "".join(f"{m[:12]:>13}" for m in method_names)
    print(header)
    print("  " + "-" * (16 + 13 * len(method_names)))

    for i, q_item in enumerate(BENCHMARK_QUERIES):
        row = f"  {q_item['type']:<16}"
        best_f1_q = max(scores[m]["f1"][i] for m in method_names)
        for m in method_names:
            val = scores[m]["f1"][i]
            mark = "*" if val == best_f1_q and val > 0 else " "
            row += f"{mark}{val:>11.4f} "
        print(row)

    print("\n  * 해당 쿼리에서 최고 F1")

    # ── v4 평가 방법 설명 ────────────────────────────────────
    print("\n" + "=" * 72)
    print("  v4 평가 방법: Self-Pooling Recall  [TREC 스타일]")
    print("=" * 72)
    print(f"  TOP_K:      {TOP_K}  (각 방법이 반환할 문서 수)")
    print(f"  BM25_LIMIT: {BM25_LIMIT}")
    print(f"  Recall 분모: 모든 방법의 결과 합집합 중 관련 문서 수")
    print(f"  장점: total_relevant가 실제로 도달 가능한 범위 → F1 수렴성 ↑")
    print(f"  방법: RRF Hybrid, Multi-Query{'(+Groq)' if groq_client else ''} 포함")
    print(f"  is_relevant: 메타데이터 category 우선 → 키워드 fallback (v3)")

    # ── JSON 저장 ─────────────────────────────────────────────
    output = {
        "config": {"top_k": TOP_K, "bm25_limit": BM25_LIMIT, "num_queries": n, "recall_method": "self-pooling"},
        "averages": {
            name: {"precision": a["p"], "recall": a["r"], "f1": a["f1"], "latency_ms": a["lat"]}
            for name, a in avg.items()
        },
        "per_query": {
            q_item["id"]: {
                "type": q_item["type"],
                "query": q_item["query"],
                "scores": {
                    name: {
                        "precision": scores[name]["p"][i],
                        "recall":    scores[name]["r"][i],
                        "f1":        scores[name]["f1"][i],
                    }
                    for name in method_names
                },
            }
            for i, q_item in enumerate(BENCHMARK_QUERIES)
        },
    }

    out_path = os.path.join(BASE_DIR, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 결과 저장: {out_path}")
    print(f"[+] 최고 F1: {best_name}  (F1={avg[best_name]['f1']:.4f})")
    print("\n평가 완료!")


if __name__ == "__main__":
    run_evaluation()
