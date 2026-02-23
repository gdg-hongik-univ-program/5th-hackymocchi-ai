"""
HackyMocchi — Penetration Test Report PDF Generator
Professional pentest report layout using ReportLab Platypus
"""
import io
import os
import re
import html
from datetime import datetime, timezone, timedelta

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, NextPageTemplate, PageBreak,
)

# ── Font Registration ──────────────────────────────────────────────
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_fonts_registered = False

def _register_fonts():
    global _fonts_registered
    if _fonts_registered:
        return
    pdfmetrics.registerFont(TTFont("NotoR", os.path.join(FONT_DIR, "NotoSansKR-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("NotoB", os.path.join(FONT_DIR, "NotoSansKR-Bold.ttf")))
    _fonts_registered = True

# ── Color Palette ──────────────────────────────────────────────────
C_NAVY    = colors.HexColor("#0D1B2A")   # deep navy
C_BLUE    = colors.HexColor("#1565C0")   # accent blue
C_RED     = colors.HexColor("#B71C1C")   # critical red
C_ORANGE  = colors.HexColor("#E65100")   # high orange
C_YELLOW  = colors.HexColor("#F9A825")   # medium yellow
C_GREEN   = colors.HexColor("#1B5E20")   # low green
C_LIGHT   = colors.HexColor("#F4F6F9")   # light bg
C_BORDER  = colors.HexColor("#C5CAD3")   # border grey
C_TEXT    = colors.HexColor("#1A1A2E")   # body text
C_DIM     = colors.HexColor("#6B7280")   # dim text
C_WHITE   = colors.white
C_SUCCESS = colors.HexColor("#1B5E20")
C_FAIL    = colors.HexColor("#B71C1C")

W, H = A4   # 595 x 842 pt

# ── Risk mapping ───────────────────────────────────────────────────
RISK_MAP = {
    "ctf":       ("MEDIUM",   C_YELLOW,  "CVSS 5.3"),
    "sqli_auth": ("CRITICAL", C_RED,     "CVSS 9.8"),
    "xss":       ("HIGH",     C_ORANGE,  "CVSS 7.4"),
    "xssi":      ("MEDIUM",   C_YELLOW,  "CVSS 6.1"),
    "lfi":       ("CRITICAL", C_RED,     "CVSS 9.1"),
    "idor":      ("HIGH",     C_ORANGE,  "CVSS 7.5"),
    "recon":     ("MEDIUM",   C_YELLOW,  "CVSS 5.3"),
    "sqli":      ("CRITICAL", C_RED,     "CVSS 9.8"),
    "generic":   ("HIGH",     C_ORANGE,  "CVSS 7.0"),
    "none":      ("NONE",     C_GREEN,   "N/A"),
}

TYPE_NAME_MAP = {
    "ctf":       "Sensitive Data Exposure (CTF Mission)",
    "sqli_auth": "SQL Injection — Auth Bypass",
    "xss":       "Cross-Site Scripting (XSS)",
    "xssi":      "Cross-Site Script Inclusion (XSSI)",
    "lfi":       "Local File Inclusion / Path Traversal",
    "idor":      "Insecure Direct Object Reference (IDOR)",
    "recon":     "Information Disclosure",
    "sqli":      "SQL Injection",
    "generic":   "Web Application Vulnerability",
    "none":      "N/A",
}


# ── Attack Type Detection (shared, importable) ─────────────────────
def _detect_attack_type(state: dict) -> str:
    """Infer attack type from URL and discovered indicators."""
    # Support both underscore-prefixed (server) and plain (frontend) field names
    url = (state.get("_attack_url") or state.get("attack_url", "")).lower()
    ind_list = state.get("_indicators_found") or state.get("indicators_found", [])
    indicators = " ".join(ind_list)

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
        return "sqli"
    # ── Gruyere / XSSI ───────────────────────────────────────────────
    if "feed.gtl" in url or "snippets.gtl" in url:
        return "xssi"
    if "gruyere" in url and ("<script>" in url or "onerror" in url or "alert(" in url):
        return "xss"
    # ── Standard types ────────────────────────────────────────────────
    if "/missions/" in url or "[HIDDEN INPUT]" in indicators or "[JS VAR]" in indicators:
        return "ctf"
    if state.get("_jwt_token") or state.get("jwt_token") or "jwt token captured" in indicators.lower():
        return "sqli_auth"
    if "<script>" in url or "alert(" in url or "xss" in url:
        return "xss"
    if "etc/passwd" in url or "../" in url or "root:x:0:0:" in indicators:
        return "lfi"
    if re.search(r'/api/users?', url) and state.get("http_method", state.get("attack_method", "")) == "GET":
        return "idor"
    if any(x in url for x in ["robots.txt", ".htpasswd", ".env", ".bak", "/admin"]):
        return "recon"
    if any(x in url for x in ["or 1=1", "or true", "union select", "' or"]) or "SQL syntax" in indicators:
        return "sqli"
    return "generic"


# ── Style Helpers ──────────────────────────────────────────────────
def S(name, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)

def make_styles():
    base = dict(fontName="NotoR", leading=16, textColor=C_TEXT)
    bold = dict(fontName="NotoB", leading=16, textColor=C_TEXT)
    return {
        "cover_title": S("cover_title", fontName="NotoB", fontSize=28,
                         textColor=C_WHITE, leading=36, alignment=TA_CENTER),
        "cover_sub":   S("cover_sub",   fontName="NotoR", fontSize=13,
                         textColor=colors.HexColor("#B0BEC5"), leading=20, alignment=TA_CENTER),
        "cover_label": S("cover_label", fontName="NotoR", fontSize=9,
                         textColor=colors.HexColor("#90A4AE"), leading=14, alignment=TA_CENTER),
        "section_h":   S("section_h",   fontName="NotoB", fontSize=13,
                         textColor=C_NAVY, leading=20, spaceAfter=6,
                         borderPadding=(0, 0, 4, 0)),
        "body":        S("body",        **base, fontSize=10, spaceAfter=4),
        "body_bold":   S("body_bold",   **bold, fontSize=10, spaceAfter=4),
        "small":       S("small",       fontName="NotoR", fontSize=8,
                         textColor=C_DIM, leading=12),
        "code":        S("code",        fontName="Courier", fontSize=8,
                         textColor=C_TEXT, leading=12, backColor=C_LIGHT,
                         borderPadding=4),
        "tbl_hdr":     S("tbl_hdr",    fontName="NotoB", fontSize=9,
                         textColor=C_WHITE, leading=14, alignment=TA_CENTER),
        "tbl_cell":    S("tbl_cell",   fontName="NotoR", fontSize=9,
                         textColor=C_TEXT, leading=14),
        "tbl_cell_b":  S("tbl_cell_b", fontName="NotoB", fontSize=9,
                         textColor=C_TEXT, leading=14),
        "risk_badge":  S("risk_badge",  fontName="NotoB", fontSize=16,
                         textColor=C_WHITE, leading=22, alignment=TA_CENTER),
        "risk_cvss":   S("risk_cvss",   fontName="NotoR", fontSize=9,
                         textColor=C_WHITE, leading=14, alignment=TA_CENTER),
        "footer":      S("footer",      fontName="NotoR", fontSize=7,
                         textColor=C_DIM, leading=10, alignment=TA_CENTER),
        "analysis":    S("analysis",    fontName="NotoR", fontSize=10,
                         textColor=C_TEXT, leading=16, spaceAfter=4,
                         leftIndent=6, borderPadding=(6, 8, 6, 8),
                         backColor=colors.HexColor("#EEF2FF")),
    }

# ── Page Templates ─────────────────────────────────────────────────
def _cover_frame():
    return Frame(0, 0, W, H, leftPadding=0, rightPadding=0,
                 topPadding=0, bottomPadding=0, id="cover")

def _body_frame():
    return Frame(2.0*cm, 2.2*cm, W-4.0*cm, H-3.8*cm,
                 leftPadding=0, rightPadding=0,
                 topPadding=0, bottomPadding=0, id="body")

class ReportDoc(BaseDocTemplate):
    def __init__(self, buf, date_str, target_url):
        super().__init__(buf, pagesize=A4,
                         rightMargin=0, leftMargin=0,
                         topMargin=0, bottomMargin=0)
        self.date_str = date_str
        self.target_url = target_url

        cover_tpl  = PageTemplate(id="cover", frames=[_cover_frame()])
        body_tpl   = PageTemplate(id="body",  frames=[_body_frame()],
                                  onPage=self._body_page)
        self.addPageTemplates([cover_tpl, body_tpl])

    def _body_page(self, canvas, doc):
        canvas.saveState()
        # Top bar
        canvas.setFillColor(C_NAVY)
        canvas.rect(0, H-1.1*cm, W, 1.1*cm, fill=1, stroke=0)
        canvas.setFont("NotoB", 8)
        canvas.setFillColor(C_WHITE)
        canvas.drawString(2.0*cm, H-0.72*cm, "HackyMocchi — Penetration Test Report")
        canvas.setFont("NotoR", 7)
        canvas.setFillColor(colors.HexColor("#B0BEC5"))
        canvas.drawRightString(W-2.0*cm, H-0.72*cm, f"CONFIDENTIAL  |  {self.date_str}")

        # Bottom bar
        canvas.setFillColor(C_NAVY)
        canvas.rect(0, 0, W, 1.2*cm, fill=1, stroke=0)
        canvas.setFont("NotoR", 7)
        canvas.setFillColor(colors.HexColor("#90A4AE"))
        trunc = self.target_url if len(self.target_url) < 60 else self.target_url[:57] + "..."
        canvas.drawString(2.0*cm, 0.42*cm, f"Target: {trunc}")
        canvas.setFont("NotoR", 7)
        canvas.drawRightString(W-2.0*cm, 0.42*cm, f"Page {doc.page - 1}")
        canvas.restoreState()

# ── Cover Page Builder ─────────────────────────────────────────────
def _cover_page(st: dict, styles: dict, atype: str, risk_color) -> list:
    elems = []
    risk_label, risk_color, cvss = RISK_MAP.get(atype, RISK_MAP["generic"])
    is_success = st.get("is_success", False)
    result_color = C_SUCCESS if is_success else C_FAIL

    def draw_cover(canvas, doc):
        canvas.saveState()
        # Background
        canvas.setFillColor(C_NAVY)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        # Accent stripe left
        canvas.setFillColor(C_BLUE)
        canvas.rect(0, 0, 0.6*cm, H, fill=1, stroke=0)
        # Top accent line
        canvas.setFillColor(C_BLUE)
        canvas.rect(0.6*cm, H-0.4*cm, W-0.6*cm, 0.4*cm, fill=1, stroke=0)

        # CONFIDENTIAL watermark (rotated, faint)
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#1E2D42"))
        canvas.setFont("NotoB", 64)
        canvas.translate(W/2, H/2)
        canvas.rotate(35)
        canvas.drawCentredString(0, 0, "CONFIDENTIAL")
        canvas.restoreState()

        # Logo text (top-left)
        canvas.setFont("NotoB", 11)
        canvas.setFillColor(colors.HexColor("#64B5F6"))
        canvas.drawString(1.4*cm, H-1.8*cm, "HackyMocchi")
        canvas.setFont("NotoR", 8)
        canvas.setFillColor(colors.HexColor("#78909C"))
        canvas.drawString(1.4*cm, H-2.4*cm, "Automated Penetration Testing Platform")

        # Date & Classification (top-right)
        canvas.setFont("NotoR", 9)
        canvas.setFillColor(colors.HexColor("#90A4AE"))
        canvas.drawRightString(W-1.4*cm, H-1.8*cm, doc.date_str if hasattr(doc, 'date_str') else "")
        canvas.setFont("NotoB", 9)
        canvas.setFillColor(colors.HexColor("#EF9A9A"))
        canvas.drawRightString(W-1.4*cm, H-2.4*cm, "CONFIDENTIAL")

        # Divider
        canvas.setStrokeColor(C_BLUE)
        canvas.setLineWidth(1)
        canvas.line(1.4*cm, H-3.0*cm, W-1.4*cm, H-3.0*cm)

        # Main title
        canvas.setFont("NotoB", 30)
        canvas.setFillColor(C_WHITE)
        canvas.drawCentredString(W/2, H-5.2*cm, "모의침투 테스트 보고서")
        canvas.setFont("NotoR", 14)
        canvas.setFillColor(colors.HexColor("#B0BEC5"))
        canvas.drawCentredString(W/2, H-6.4*cm, "Penetration Test Report")

        # Result badge
        badge_y = H - 9.5*cm
        badge_w = 7*cm
        badge_x = (W - badge_w) / 2
        result_text = "VULNERABILITY FOUND" if is_success else "NO VULNERABILITY FOUND"
        canvas.setFillColor(result_color)
        canvas.roundRect(badge_x, badge_y, badge_w, 1.2*cm, 6, fill=1, stroke=0)
        canvas.setFont("NotoB", 11)
        canvas.setFillColor(C_WHITE)
        canvas.drawCentredString(W/2, badge_y + 0.38*cm, result_text)

        # Risk badge
        risk_badge_y = badge_y - 2.2*cm
        risk_w = 4.5*cm
        risk_x = (W - risk_w) / 2
        canvas.setFillColor(risk_color)
        canvas.roundRect(risk_x, risk_badge_y, risk_w, 1.6*cm, 6, fill=1, stroke=0)
        canvas.setFont("NotoB", 20)
        canvas.setFillColor(C_WHITE)
        canvas.drawCentredString(W/2, risk_badge_y + 0.7*cm, risk_label)
        canvas.setFont("NotoR", 8)
        canvas.setFillColor(colors.HexColor("#FFCCBC"))
        canvas.drawCentredString(W/2, risk_badge_y + 0.2*cm, cvss)

        # Info table (cover)
        info_y = risk_badge_y - 0.8*cm
        rows = [
            ("대상 URL", st.get("target_url", "N/A")),
            ("대상 IP", st.get("target_ip", "Unknown")),
            ("서버 / 기술 스택", f"{st.get('server','Unknown')}  /  {st.get('detected_tech','Unknown')}"),
            ("총 시도 횟수", f"{st.get('attempts', 0)} 회"),
            ("테스트 일시", doc.date_str if hasattr(doc, 'date_str') else ""),
        ]
        lbl_w = 4.0*cm
        val_w = W - 2.8*cm - lbl_w - 1.0*cm
        row_h = 0.75*cm
        for i, (lbl, val) in enumerate(rows):
            ry = info_y - i * row_h
            # Row bg alternate
            bg = colors.HexColor("#142032") if i % 2 == 0 else colors.HexColor("#0F1A28")
            canvas.setFillColor(bg)
            canvas.rect(1.4*cm, ry - 0.55*cm, W-2.8*cm, row_h, fill=1, stroke=0)
            canvas.setFont("NotoB", 8)
            canvas.setFillColor(colors.HexColor("#90A4AE"))
            canvas.drawString(1.8*cm, ry - 0.28*cm, lbl)
            canvas.setFont("NotoR", 8)
            canvas.setFillColor(C_WHITE)
            val_str = val if len(str(val)) < 70 else str(val)[:67] + "..."
            canvas.drawString(1.8*cm + lbl_w + 0.5*cm, ry - 0.28*cm, str(val_str))

        # Footer
        canvas.setFont("NotoR", 7)
        canvas.setFillColor(colors.HexColor("#546E7A"))
        canvas.drawCentredString(W/2, 1.0*cm,
            "본 보고서는 허가된 환경에서 수행된 모의침투 테스트 결과입니다. 무단 배포를 금합니다.")
        canvas.restoreState()

    elems.append(Spacer(1, H))  # placeholder — actual drawing via onFirstPage
    return elems, draw_cover


# ── Section Header ─────────────────────────────────────────────────
def sec_header(title: str, styles: dict) -> list:
    return [
        HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=6),
        Paragraph(title, styles["section_h"]),
        Spacer(1, 4),
    ]

def kv_table(rows: list, styles: dict) -> Table:
    """Two-column key-value table."""
    data = [[
        Paragraph(k, styles["tbl_cell_b"]),
        Paragraph(str(v), styles["tbl_cell"]),
    ] for k, v in rows]
    t = Table(data, colWidths=[4.5*cm, None])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), C_LIGHT),
        ("GRID",       (0, 0), (-1, -1), 0.5, C_BORDER),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [C_WHITE, colors.HexColor("#F9FAFB")]),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    return t


# ── Main Entry Point ───────────────────────────────────────────────
def build_pdf(state: dict, atype: str) -> bytes:
    """Generate professional pentest PDF report. Returns bytes."""
    _register_fonts()

    buf = io.BytesIO()
    KST = timezone(timedelta(hours=9))
    date_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M (KST)")
    styles = make_styles()

    risk_label, risk_color, cvss = RISK_MAP.get(atype, RISK_MAP["generic"])
    type_name = TYPE_NAME_MAP.get(atype, "Web Application Vulnerability")
    is_success = state.get("is_success", False)

    doc = ReportDoc(buf, date_str, state.get("target_url", ""))

    story = []

    # ── Page 1: Cover ───────────────────────────────────────────────
    cover_elems, draw_cover = _cover_page(state, styles, atype, risk_color)

    class CoverPage(Spacer):
        def draw(self):
            draw_cover(self.canv, self.canv._doctemplate)

    story.append(CoverPage(1, H))
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    # ── Page 2+: Body ───────────────────────────────────────────────
    # 1. Executive Summary
    story += sec_header("1. Executive Summary", styles)

    exec_rows = [
        ("결과",          "VULNERABILITY FOUND" if is_success else "NO VULNERABILITY FOUND"),
        ("공격 유형",     type_name),
        ("위험도",        f"{risk_label}  ({cvss})"),
        ("대상 URL",      state.get("target_url", "N/A")),
        ("대상 IP",       state.get("target_ip", "Unknown")),
        ("기술 스택",     state.get("detected_tech", "Unknown")),
        ("총 시도 횟수",  f"{state.get('attempts', 0)} 회"),
        ("테스트 일시",   date_str),
    ]
    story.append(kv_table(exec_rows, styles))
    story.append(Spacer(1, 14))

    # 2. Vulnerability Details
    story += sec_header("2. 취약점 상세", styles)

    indicators = state.get("indicators_found", state.get("_indicators_found", []))
    attack_url  = state.get("attack_url",  state.get("_attack_url", state.get("final_payload", "")))
    attack_method = state.get("attack_method", state.get("_attack_method", state.get("http_method", "GET")))
    attack_data   = state.get("attack_data",   state.get("_attack_data",  state.get("post_data", {})))
    jwt_token     = state.get("jwt_token",     state.get("_jwt_token", ""))
    cap_email     = state.get("captured_email",state.get("_captured_email", ""))
    cap_role      = state.get("captured_role", state.get("_captured_role", ""))

    detail_rows = [
        ("취약 유형",        type_name),
        ("HTTP 메소드",      attack_method),
        ("공격 URL",         attack_url or "N/A"),
        ("POST 데이터",      str(attack_data) if attack_data else "없음"),
        ("발견된 지표",      ", ".join(indicators) if indicators else "없음"),
    ]
    if jwt_token:
        detail_rows.append(("JWT Token", jwt_token[:80] + ("..." if len(jwt_token) > 80 else "")))
    if cap_email:
        detail_rows.append(("캡처된 이메일", cap_email))
    if cap_role:
        detail_rows.append(("캡처된 권한", cap_role))

    story.append(kv_table(detail_rows, styles))
    story.append(Spacer(1, 14))

    # 3. Risk Analysis
    story += sec_header("3. 결과 분석", styles)

    ANALYSIS = {
        "ctf": (
            "페이지 소스에 인증 정보가 노출되어 있습니다. hidden input 필드 또는 JS 변수에 "
            "패스워드가 평문으로 포함되어 있어 소스 보기만으로 누구든 미션을 통과할 수 있습니다. "
            "이는 클라이언트 측 보안에 의존하는 설계 결함으로, 민감한 데이터를 절대 HTML 소스에 "
            "포함해서는 안 됩니다."
        ),
        "sqli_auth": (
            "SQL Injection을 통한 인증 우회에 성공하여 JWT 토큰이 탈취되었습니다. 입력값이 SQL 쿼리에 "
            "직접 삽입되고 있으며, 공격자는 패스워드 없이 임의 계정(관리자 포함)으로 로그인할 수 있습니다. "
            "이 취약점은 OWASP Top 10 A03:2021 Injection에 해당하며 즉각적인 조치가 필요합니다."
        ),
        "xss": (
            "사용자 입력값이 HTML에 그대로 출력되어 스크립트 삽입이 가능합니다. 공격자는 피해자 "
            "브라우저에서 임의 코드를 실행하거나 세션 쿠키를 탈취할 수 있습니다. "
            "이는 OWASP Top 10 A03:2021 Injection에 해당하며 모든 출력에 이스케이핑이 필요합니다."
        ),
        "xssi": (
            "Cross-Site Script Inclusion(XSSI) 취약점이 확인되었습니다. feed.gtl 등 JSON 데이터를 "
            "JavaScript 함수 호출 형태로 반환하는 엔드포인트가 교차 출처 <script> 태그로 포함 가능합니다. "
            "공격자가 제어하는 페이지에서 피해자의 인증된 스니펫·개인정보를 탈취할 수 있으며, "
            "이는 OWASP Top 10 A02:2021 Cryptographic Failures / 정보 노출에 해당합니다."
        ),
        "lfi": (
            "경로 탐색(Path Traversal) 취약점으로 서버 내부 파일 읽기에 성공했습니다. "
            "공격자는 /etc/passwd, SSH 키, 소스코드, 설정파일 등을 읽을 수 있습니다. "
            "이는 OWASP Top 10 A01:2021 Broken Access Control에 해당합니다."
        ),
        "idor": (
            "접근 제어가 없는 API 엔드포인트에서 타 사용자 데이터 조회에 성공했습니다. "
            "ID 값 조작만으로 모든 계정의 개인정보에 접근할 수 있으며, 이는 "
            "OWASP Top 10 A01:2021 Broken Access Control에 해당합니다."
        ),
        "recon": (
            "공개된 파일/경로를 통해 내부 구조 정보가 노출되었습니다. "
            "robots.txt, 백업 파일, 숨겨진 관리자 경로 등이 외부에서 접근 가능하며, "
            "수집된 정보는 추가 공격의 진입점으로 활용될 수 있습니다."
        ),
        "sqli": (
            "SQL Injection 취약점이 확인되었습니다. 입력값이 SQL 쿼리에 직접 삽입되어 "
            "DB 데이터 열람 및 인증 우회가 가능합니다. "
            "이는 OWASP Top 10 A03:2021 Injection에 해당하며 즉각적인 조치가 필요합니다."
        ),
        "generic": "취약점이 확인되었습니다. 발견된 성공 지표를 기반으로 상세 분석이 필요합니다.",
        "none":    ("현재 설정으로는 취약점을 확인하지 못했습니다. 더 정교한 페이로드가 필요하거나 "
                    "대상이 WAF 등 방어 기법을 적용하고 있을 수 있습니다. "
                    "정기적인 모의침투 테스트와 취약점 스캔을 권장합니다."),
    }
    analysis_text = ANALYSIS.get(atype, "취약점이 확인되었습니다.")
    # ReportLab Paragraph는 HTML 유사 마크업을 파싱하므로
    # <script> 같은 문자열은 escape해서 렌더 오류를 방지한다.
    story.append(Paragraph(html.escape(analysis_text), styles["analysis"]))
    story.append(Spacer(1, 14))

    # 4. Remediation
    story += sec_header("4. 권장 조치사항", styles)

    REMEDIATION = {
        "ctf":       ["서버 측 패스워드 파일에 대한 외부 접근 차단 (웹 루트 외부에 저장)",
                      "HTML 소스에 인증 정보(hidden field, JS 변수)를 절대 포함하지 않기",
                      "인증 로직은 반드시 서버 사이드에서만 처리"],
        "sqli_auth": ["Prepared Statements (매개변수화된 쿼리) 사용",
                      "ORM 사용으로 직접 SQL 조합 제거",
                      "입력값 유효성 검사 — 특수문자 필터링",
                      "에러 메시지에 SQL 정보 노출 금지",
                      "DB 계정 최소 권한 원칙 적용"],
        "xssi":      ["JSON 응답 앞에 )]}', 또는 while(1); prefix 삽입으로 JSON hijacking 방지",
                      "동적 데이터를 JavaScript 함수 호출 형태(JSONP)로 반환하지 말 것",
                      "민감 API에 CORS 정책 적용 — 허용된 출처만 접근 가능하도록 설정",
                      "응답 Content-Type을 application/json으로 설정 (text/javascript 금지)",
                      "모든 민감 엔드포인트에 CSRF 토큰 및 인증 검증 추가"],
        "xss":       ["모든 출력값에 HTML 이스케이핑 적용 (htmlspecialchars 등)",
                      "Content-Security-Policy (CSP) 헤더 설정",
                      "HttpOnly / Secure 쿠키 플래그 설정으로 쿠키 탈취 방지",
                      "입력값 화이트리스트 기반 검증"],
        "lfi":       ["파일 경로에 사용자 입력값 직접 사용 금지",
                      "허용된 파일 목록(화이트리스트)만 접근 허용",
                      "open_basedir 설정으로 웹 루트 외부 접근 차단",
                      "입력값에서 ../ 시퀀스 필터링"],
        "idor":      ["모든 API 요청에 인증 및 권한 검사 적용",
                      "리소스 접근 시 소유권 검증 (현재 로그인 사용자 소유 여부)",
                      "순차적 ID 대신 UUID 사용으로 열거 공격 방지"],
        "recon":     ["robots.txt에 민감한 경로 노출 금지",
                      "불필요한 파일 (.htpasswd, .env, .bak) 웹 루트에서 제거",
                      "디렉토리 리스팅 비활성화 (Options -Indexes)",
                      "민감한 파일에 대한 웹 서버 수준 접근 제한"],
        "sqli":      ["Prepared Statements (매개변수화된 쿼리) 사용",
                      "입력값 유효성 검사 및 특수문자 필터링",
                      "WAF(Web Application Firewall) 도입",
                      "에러 메시지에 SQL 정보 노출 금지",
                      "최소 권한 원칙 — DB 계정에 필요한 권한만 부여"],
        "generic":   ["입력값 검증 및 화이트리스트 기반 필터링",
                      "WAF 도입",
                      "정기적인 보안 취약점 점검 실시"],
        "none":      ["정기적인 모의해킹 및 취약점 스캔 수행",
                      "WAF 및 보안 모니터링 유지"],
    }
    remediation_list = REMEDIATION.get(atype, REMEDIATION["generic"])

    rem_data = [[
        Paragraph(f"{i+1}.", styles["tbl_cell_b"]),
        Paragraph(item, styles["tbl_cell"]),
    ] for i, item in enumerate(remediation_list)]

    rem_table = Table(rem_data, colWidths=[0.8*cm, None])
    rem_table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_WHITE, colors.HexColor("#F9FAFB")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",    (0, 0), (0, -1), C_LIGHT),
    ]))
    story.append(rem_table)
    story.append(Spacer(1, 14))

    # 5. Technical Evidence (response preview)
    resp_preview = state.get("response_preview", state.get("_response_preview", ""))
    if resp_preview and resp_preview.strip():
        story += sec_header("5. 기술적 증거 (Response Preview)", styles)
        preview_text = resp_preview[:600].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(preview_text, styles["code"]))
        story.append(Spacer(1, 14))

    # Disclaimer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "본 보고서는 HackyMocchi 플랫폼이 허가된 환경에서 수행한 모의침투 테스트 결과입니다. "
        "결과 내용은 테스트 시점을 기준으로 하며, 실제 운영 환경의 보안 상태를 완전히 반영하지 않을 수 있습니다. "
        "무단 배포 및 허가되지 않은 시스템에 대한 활용을 금합니다.",
        styles["small"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
