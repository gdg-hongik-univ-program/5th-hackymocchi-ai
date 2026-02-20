# Hackymocchi

**LLM을 이용한 보안 취약점 발견**

---

## 1. 프로젝트 개요

Hackymocchi(해키모찌)는 URL을 입력받아 웹 애플리케이션의 HTML/JavaScript 코드를 수집하고,
RAG(Retrieval-Augmented Generation) 기반 LLM을 활용해 **보안 취약점을 분석**하는 프로젝트입니다.

대규모 취약점 데이터와 익스플로잇 정보를 벡터 데이터베이스에 적재한 뒤,
이를 기반으로 **취약점 탐지 및 공격 가능성 분석**을 수행합니다.

---

## 2. Teams Members (팀원 소개)

|정승윤|문선영|신대환|이정재|
|:---:|:---:|:---:|:---:|
|[AI]|[AI]|[AI]|[AI]|

---

## 3. Task & Responsibilities (작업 및 역할 분담)

| 이름    | 역할                                  |
| ---------- | ---------------------------------------------- |
| 정승윤 | 데이터 전처리, LangGraph & Langchain 구현, RAG 성능 향상 |
| 문선영 | 데이터 전처리, Vector DB 적재, RAG 성능 향상 (Hybrid search)|
| 신대환 | 데이터 전처리, Vector DB 적재, RAG 성능 향상 (Re-ranking)|
| 이정재 | 데이터 전처리, LangGraph & Langchain 구현, UI 제작 |

---

## 4. 아키텍처 개요

```
[사용자 URL 입력]
        ↓
[LangChain 기반 웹 크롤러]
        ↓
[HTML / JS 코드 수집]
        ↓
[Vector DB (취약점/익스플로잇 데이터)]
        ↓
[RAG 검색]
        ↓
[LLM (g5xlarge)]
        ↓
[취약점 분석 결과 출력]
```

---

## 5. 데이터 소스

[NVD (National Vulnerability Database)](https://nvd.nist.gov/vuln/data-feeds) – CVE, CVSS, 공식 보안 권고

[ProjectDiscovery Nuclei](https://github.com/projectdiscovery/nuclei) – 템플릿 기반 취약점 탐지 패턴

[PayloadsAllTheThings](https://github.com/swisskyrepo/PayloadsAllTheThings) – XSS, SQLi, RCE 등 공격 페이로드 모음

[Exploit Database](https://gitlab.com/exploit-database/exploitdb) – 공개 익스플로잇 코드 및 공격 사례

각 데이터는 전처리 후 임베딩되어 **Vector DB**에 저장됩니다.

---

## 6. 데이터 처리 파이프라인

1. 취약점 데이터 전처리 - 텍스트 정규화 / 메타데이터 제거 / 취약점 유형 구조화

2. 임베딩 생성

3. Vector DB 적재

4. RAG 기반 검색 시스템 구성

---

## 7. 기술 스택

* **Language:** Python
* **Framework:** LangChain
* **Model:** g5xlarge
* **Architecture:** RAG 기반 LLM
* **Database:** Vector Database


---

## 8. 주요 기능

* URL 입력 기반 웹 크롤링
* HTML / JavaScript 코드 수집
* 취약점 유형 분석
* RAG 기반 취약점 유사도 검색
* 취약점 발견 및 공격 가능성 평가

---

## 9. 설치 방법 (Installation)

```bash
git clone https://github.com/your-repo/hackymocchi.git
cd hackymocchi
pip install -r requirements.txt
```

---

## 10. 사용 방법 (Usage)

주어진 화면에 타켓 URL 입력

실행 시:

1. 웹 크롤러가 HTML/JS 코드 수집
2. Vector DB에서 관련 취약점 검색
3. LLM이 취약점 분석 수행
4. 결과 리포트 출력

---

## 11. 윤리 및 법적 고지 (Responsible Use & Legal Notice)

### 목적

Hackymocchi는 **보안 연구 및 교육 목적**으로 개발된 프로젝트입니다.
웹 애플리케이션의 취약점을 식별하고 보안 강화를 지원하기 위한 연구용 도구입니다.

---

### 사용 제한

다음 행위는 엄격히 금지됩니다:

* 허가 없이 타인의 시스템을 스캔하거나 공격하는 행위
* 무단 침투, 서비스 방해(DoS), 데이터 탈취 시도
* 관련 법률을 위반하는 모든 행위
* 상업적 악용 또는 악의적 사용

본 프로젝트는 **명시적으로 허가된 테스트 환경에서만 사용**해야 합니다.

---

### 법적 책임

개발자는 다음에 대해 책임을 지지 않습니다:

* 사용자에 의해 발생한 법적 문제
* 무단 테스트로 인한 손해
* 시스템 침해, 데이터 유출, 서비스 중단 등 모든 피해

사용자는 해당 국가의 사이버 보안 관련 법률을 준수할 책임이 있습니다.

---

### 권장 사용 환경

* 로컬 테스트 서버
* CTF 환경
* 자체 소유 인프라
* 명시적으로 허가된 보안 테스트 환경

---

### Responsible Disclosure 권장

취약점 발견 시:

* 서비스 제공자에게 비공개로 먼저 통보
* CVE 등록 절차 준수
* 패치 기간 제공 후 공개

---

