"""알고핏 UI 설계서 만들기 → UI설계서_알고핏.docx (로고 의미 · 디자인 기준 · 화면 구조 · 화면별 설계)
실행: python3 UI설계서_만들기.py   (화면캡처/ 폴더의 그림을 사용)"""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

D = Path(__file__).resolve().parent
CAP, LOGO, OUT = D / "화면캡처", D / "로고", D / "UI설계서_그림"
OUT.mkdir(exist_ok=True)
BLUE = (61, 90, 254)
FONT = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 30, index=6)

# 화면별 번호 표시 위치 (화면 좌표: 1280×800 기준) 와 설명
SCREENS = [
    ("1_내학습리포트.jpg", "학습 리포트 (첫 화면)", [
        ((160, 28), "로고", "사이드바 맨 위에 알고핏 로고. 사이드바를 접으면 심볼만 남는다"),
        ((290, 131), "검색", "Claude처럼 맨 위 검색창. 저장한 문제를 이름으로 찾는다"),
        ((290, 173), "내 학습 리포트", "가장 중요한 진단 결과를 메뉴 맨 위에 둔다 (팀 결정)"),
        ((290, 214), "새로 만들기", "막힌 문제를 새로 저장하는 입구. 동그란 + 아이콘으로 눈에 띄게"),
        ((290, 300), "최근 문제", "저장한 문제 목록. 누르면 그 대화로 돌아간다. 옆 ⋯ 로 이름 바꾸기·해결 표시·삭제"),
        ((290, 760), "내 아이디·이름", "Claude 계정 메뉴처럼 맨 아래. 누르면 Codeforces 아이디·이름 입력"),
        ((345, 60), "제목 상자", "누구의 리포트인지, 진단 기간, 학습 유형을 한눈에"),
        ((345, 250), "핵심 숫자 4개", "실력 점수(공식) · 평소 실력(실효 레이팅) · 전체 판정 · 약점/강점 수. 물음표에 쉬운 설명"),
        ((345, 400), "약점·강점 카드", "살구색 = 보완 필요, 파랑 = 잘함. 색만이 아니라 글자로도 판정을 적어 색약 사용자도 구분"),
        ((345, 640), "분야별 그래프", "평소 실력 대비 몇 % 더/덜 맞혔는지. 막대 끝에 숫자를 함께 표시"),
    ]),
    ("2_AI코치_맞춤지도방향.jpg", "AI 코치 설명과 맞춤 지도 방향", [
        ((345, 150), "AI 코치 한마디", "진단 결과를 한두 문장으로 요약 (gpt-4o-mini, JSON 응답)"),
        ((345, 300), "맞춤 지도 방향 3칸", "무엇을 · 어느 수준에서 · 어떻게. 굵은 글씨는 진단 규칙이 정한 방향, 아래는 AI 설명"),
        ((345, 560), "보완할 점 · 잘하는 점 · 풀이 습관", "근거 숫자와 함께 3열로 비교"),
    ]),
    ("3_새로만들기.jpg", "새로 만들기", [
        ((560, 150), "인사말", "Claude 첫 화면처럼 가운데 질문 한 줄로 시작. 강조 점은 브랜드 파랑"),
        ((345, 268), "캡처로 문제 넣기", "문제 화면을 올리면 AI(gpt-4o)가 읽어 문제칸에 채움"),
        ((345, 340), "둥근 입력 카드", "문제 이름 · 분야 · 난이도 · 문제 · 내 코드. 분야·난이도로 도움 정도를 정함"),
    ]),
    ("4_문제대화_단계별힌트.jpg", "문제 풀이 대화", [
        ((345, 27), "도움 정도 한 줄", "제목 바로 아래 한 줄: 조금씩·보통·많이 선택과 진단으로 정한 추천 이유. 대화 사이에 끼지 않도록 위로 올림"),
        ((345, 83), "문제와 내 코드", "고친 코드를 저장하면 다음 질문부터 고친 코드로 답함"),
        ((345, 152), "내 질문", "학습자가 쓴 말만 그대로 보여 줌 (다른 표시를 붙이지 않음)"),
        ((345, 227), "개념 + 단계별 힌트", "답변 위에 '도움 정도 · 보통'을 작게 표시. 정답 코드 대신 개념과 단계별 힌트, 글자 단위로 바로 나옴(스트리밍)"),
        ((345, 379), "단계 열기", "한 단계씩 열어 보며 스스로 풀기"),
        ((345, 597), "답변 평가 👍👎", "Claude처럼 답변 평가. 학습자 만족도 자료로 활용"),
        ((345, 750), "관계없는 질문 안내", "인사·잡담에는 힌트 없이 안내 문장 하나만 (코드에서도 한 번 더 걸러 냄)"),
    ]),
    ("6_조금씩_힌트하나씩.jpg", "실력이 높은 학습자: 힌트 조금씩", [
        ((740, 69), "해결했어요 · 대화 내보내기", "푼 문제는 ✓ 표시, 대화는 복습 노트(.md)로 저장"),
        ((345, 117), "추천 도움 정도", "진단 결과(전체 판정·분야 약점·문제 난이도)로 '조금씩'을 추천하고 이유를 한 줄로"),
        ((345, 330), "개념 한 줄 + 첫 힌트", "'조금씩'이면 처음엔 개념 한 줄과 힌트 하나만"),
        ((345, 472), "다음 힌트 보기", "필요할 때만 다음 힌트를 하나씩 연다"),
        ((345, 610), "질문창 (맨 아래 고정)", "Claude처럼 화면 맨 아래에 고정. 대화는 위에서 아래로 쌓이고 새 답변은 맨 끝에"),
    ]),
    ("7_점점점메뉴.jpg", "⋯ 메뉴", [
        ((290, 296), "⋯ 메뉴", "이름 바꾸기 · 해결로 표시 · 대화 내용 지우기 · 문제 삭제"),
    ]),
]


def annotate(fname, marks):
    im = Image.open(CAP / fname).convert("RGB")
    s = im.width / 1280
    d = ImageDraw.Draw(im)
    for i, ((x, y), *_ ) in enumerate(marks, 1):
        cx, cy, r = x * s, y * s, 21 * s / 2 * 1.4
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BLUE, outline="white", width=int(3 * s / 2))
        d.text((cx, cy), str(i), font=FONT, fill="white", anchor="mm")
    out = OUT / fname
    im.save(out, quality=92)
    return out


def shade(cell, hexcolor):
    tcPr = cell._element.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor.lstrip("#")); tcPr.append(sh)


doc = Document()
st = doc.styles["Normal"]; st.font.name = "맑은 고딕"; st.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕"); st.font.size = Pt(10)
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Cm(2); sec.top_margin = sec.bottom_margin = Cm(1.8)


def table(rows, header, widths=None):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, v in zip(t.rows[0].cells, header):
        c.text = str(v)
    for r in rows:
        for c, v in zip(t.add_row().cells, r):
            c.text = str(v)
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    return t


B = lambda t: doc.add_paragraph(t, style="List Bullet")
doc.add_picture(str(LOGO / "algofit_logo.png"), width=Cm(6))
doc.add_heading("알고핏(AlgoFit) UI 설계서", 0)
doc.add_paragraph("코딩 문제 풀이 이력 분석과 LLM을 활용한 맞춤형 학습 지원 시스템 — 학습자가 직접 쓰는 웹 화면(Streamlit)의 설계 기준과 화면 구성")

doc.add_heading("1. 서비스 이름과 로고", 1)
table([["이름", "알고핏 (AlgoFit) = Algorithm + Fit"], ["뜻", "내 실력에 딱 맞춘(fit) 알고리즘 학습. 진단 결과에 맞춰 지도 방향과 힌트 양을 정하는 서비스의 핵심을 이름에 담음"],
       ["소개 문구", "내 실력에 딱 맞춘 알고리즘 학습 코치"]], ["항목", "내용"], [3, 13])
doc.add_paragraph()
doc.add_picture(str(LOGO / "algofit_mark_512.png"), width=Cm(3))
doc.add_heading("로고의 의미", 2)
B("계단 3칸: 정답을 한 번에 주지 않고 단계별 힌트로 한 칸씩 스스로 올라가는 학습 방식 (조금씩 · 보통 · 많이 세 단계의 도움 정도)")
B("맨 위의 점: 내 실력에 맞는 목표. 계단을 올라 목표에 '딱 맞게(fit)' 닿는다는 뜻")
B("둥근 사각형: 앱 아이콘처럼 어디서나 알아보기 쉬운 모양 (브라우저 탭 아이콘, 사이드바를 접었을 때의 표시)")
B("파랑(#3D5AFE): 신뢰·집중을 주는 색으로, 오렌지(Claude)·검정(ChatGPT)과 겹치지 않는 알고핏만의 색")
table([["algofit_logo.png / .svg", "가로형 로고 (심볼 + 알고핏 + ALGOFIT). 사이드바 맨 위, Word 리포트 맨 위, 발표 자료"],
       ["algofit_mark_512.png / .svg", "심볼만. 앱 아이콘, 접힌 사이드바, 발표 표지"], ["algofit_mark_64.png", "브라우저 탭 아이콘"]], ["파일 (로고 폴더)", "쓰는 곳"], [5.5, 10.5])
doc.add_paragraph("사용 규칙: 흰 바탕 위에 쓰고, 로고 둘레에 심볼 높이의 ¼ 이상 여백을 둔다. 비율을 바꾸거나 색을 바꾸지 않는다.")

doc.add_heading("2. 디자인 기준", 1)
doc.add_paragraph("레이아웃은 Claude의 사이드바 구조를 참고하되, 색은 흰색에 가까운 무채색 바탕 + 알고핏 파랑으로 차별화했다. 모든 글자는 배경 대비 4.5:1 이상(웹 접근성 WCAG AA)을 지킨다.")
colors = [("브랜드 파랑", "#3D5AFE", "버튼, 선택 표시, 강조 글자, 로고", "흰 글자 5.1:1"),
          ("본문 글자", "#1F1F1F", "제목, 본문, 핵심 숫자", "흰 바탕 16.1:1"),
          ("보조 글자", "#666A70", "설명, 날짜, 구간 이름", "흰 바탕 5.4:1"),
          ("바탕", "#FFFFFF", "메인 화면", "-"),
          ("사이드바", "#F9F9FA", "왼쪽 메뉴 바탕", "보조 글자 5.2:1"),
          ("상자", "#F7F7F8", "제목 상자, 입력칸", "보조 글자 5.1:1"),
          ("선택된 메뉴", "#ECEDF0", "지금 보고 있는 메뉴·문제", "본문 글자 13:1"),
          ("보완 필요", "#F4A27A / 글자 #A9512A", "약점 막대·카드", "글자 5.0:1"),
          ("잘함", "#6F98EC / 글자 #3558C9", "강점 막대·카드", "글자 5.6:1")]
t = table([[a, b, c, d] for a, b, c, d in colors], ["이름", "색", "쓰는 곳", "대비"], [3, 4, 6, 3])
for row, (_, hexc, _, _) in zip(t.rows[1:], colors):
    shade(row.cells[1], hexc.split(" ")[0])
    if hexc.startswith("#3D5AFE") or hexc.startswith("#1F1F1F") or hexc.startswith("#666A70"):
        for p in row.cells[1].paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(255, 255, 255)
doc.add_paragraph()
table([["글꼴", "Noto Sans KR (웹), 맑은 고딕(Word 리포트)"], ["글자 크기", "제목 2rem · 1.5rem · 1.25rem, 본문 1rem, 핵심 숫자 1.75rem(28px)"],
       ["모서리", "기본 10px, 입력 카드·질문창 18~20px (부드러운 인상)"], ["버튼 크기", "사이드바 메뉴 높이 40px 이상 (누르기 쉬운 크기)"],
       ["아이콘", "운영체제마다 모양이 다른 이모지 대신 Material 아이콘으로 통일"],
       ["강조 방식", "색만으로 뜻을 전하지 않고 글자(보완 필요/잘함)·숫자를 함께 표시"]], ["항목", "기준"], [3, 13])

doc.add_heading("3. 화면 구조와 사용 흐름", 1)
table([["왼쪽 사이드바", "로고 → 소개 문구 → 검색 → 내 학습 리포트 → 새로 만들기 → 최근 문제(⋯ 메뉴) → (맨 아래) 내 아이디·이름"],
       ["가운데 화면", "선택한 메뉴에 따라 '학습 리포트' 또는 '문제 풀이 도움'"]], ["영역", "구성"], [3, 13])
doc.add_paragraph("사용 흐름: ① 맨 아래에서 Codeforces 아이디 입력 → ② 학습 리포트로 내 진단·맞춤 지도 방향 확인 → ③ 막힌 문제가 있으면 새로 만들기 → "
                  "④ 문제·코드 저장 → ⑤ 질문하면 진단에 맞춘 양의 개념 설명·단계별 힌트 → ⑥ 해결 표시·내보내기, 다음에 목록에서 이어서 질문")
table([["Claude에서 참고한 점", "사이드바 구조(검색 → 새로 만들기 → 최근 목록 → 맨 아래 계정), 첫 화면 가운데 인사말, 화면 맨 아래 고정된 둥근 질문 입력창, ⋯ 메뉴, 답변 평가·다시 받기"],
       ["알고핏만의 점", "맨 위 '내 학습 리포트'(진단 결과), 진단으로 정한 도움 정도(조금씩·보통·많이)와 이유 표시, 단계별로 열어 보는 힌트, 해결 표시·복습 노트 내보내기, 관계없는 질문 안내, 파랑 브랜드 색과 계단 로고"]],
      ["구분", "내용"], [4, 12])

doc.add_heading("4. 화면별 설계", 1)
for fname, title, marks in SCREENS:
    doc.add_heading(title, 2)
    doc.add_picture(str(annotate(fname, marks)), width=Cm(16.5))
    table([[i, name, desc] for i, (_, name, desc) in enumerate(marks, 1)], ["번호", "요소", "설계 이유"], [1.2, 4, 11.3])
    doc.add_paragraph()

doc.add_heading("5. UX 원칙", 1)
for tline in ["정답 대신 길: 정답 코드를 주지 않고 개념과 단계별 힌트로 스스로 풀게 한다",
              "근거 보여 주기: 추천 도움 정도·지도 방향마다 진단 근거(판정·분야·난이도)를 함께 보여 준다",
              "쉬운 말: '레이팅'은 '실력 점수', '해결률'은 '맞힌 비율'처럼 학습자가 아는 말로 쓴다",
              "접근성: 글자 대비 4.5:1 이상, 색과 글자 함께 표시, 누르기 쉬운 버튼 크기, 휴대폰 화면에서도 한 줄씩 쌓이는 배치",
              "실수 방지: 빈 칸 저장 안내, 관계없는 질문 안내, 다시 답변 받기에 실패해도 원래 답변 유지",
              "이어서 쓰기: 저장한 문제와 대화는 앱을 다시 켜도 남고, 내보내기로 복습 노트를 만든다"]:
    B(tline)
out = D / "UI설계서_알고핏.docx"
doc.save(out)
print("저장:", out)
