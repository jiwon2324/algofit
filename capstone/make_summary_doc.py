"""분석 결과 요약 문서 (팀 공유·PPT용) — python3 make_summary_doc.py"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

out = Path("output"); fig = out / "그림"
S = json.loads(Path("model/model_settings.json").read_text(encoding="utf-8"))
D = pd.read_csv(out / "step3_diagnosis_users.csv"); CB = pd.read_csv(out / "step3_diagnosis_categories.csv")
X = pd.read_csv(out / "step4_behavior_types.csv")
v1 = pd.read_csv(out / "step6_①판정별.csv", index_col=0); v2 = pd.read_csv(out / "step6_①같은개수.csv", index_col=0)
v3 = pd.read_csv(out / "step6_②예측.csv", index_col=0); v4 = pd.read_csv(out / "step6_③레이팅.csv", index_col=0)
keep = pd.read_csv(out / "step6_④유형유지.csv", index_col=0)
FONT, NAVY, MUTED, HEAD = "맑은 고딕", "22336B", "5B6676", "EEF3FE"

doc = Document()
st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(10.5)
st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(3)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(1.8); s.top_margin = s.bottom_margin = Cm(1.8)


def run(p, t, size=10.5, bold=False, color="1E2A3A"):
    r = p.add_run(t); r.font.name = FONT; r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = RGBColor.from_string(color)


def para(t="", size=10.5, bold=False, color="1E2A3A", before=0, after=3):
    p = doc.add_paragraph()
    for i, x in enumerate(t.split("**")):
        if x:
            run(p, x, size, bold or i % 2 == 1, color)
    p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(after)
    return p


def h1(t): para(t, 15, True, NAVY, 12, 5)
def h2(t): para(t, 12, True, NAVY, 6, 3)
def bullet(t): p = para("• " + t); p.paragraph_format.left_indent = Cm(0.4)


def shade(c, f):
    pr = c._element.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), f); pr.append(sh)


def table(header, rows, widths, size=9):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; run(c.paragraphs[0], h, size, True, NAVY); shade(c, HEAD)
    for row in rows:
        cs = t.add_row().cells
        for i, v in enumerate(row):
            run(cs[i].paragraphs[0], str(v), size)
    for i, col in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
        col.set(qn("w:w"), str(int(Cm(widths[i]).twips)))
    for r in t.rows:
        for i, w in enumerate(widths):
            r.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def image(name, w=17):
    doc.add_picture(str(fig / name), width=Cm(w))


para("진단 알고리즘 분석 결과", 22, True, NAVY, after=0)
para("팀이 정한 기준으로 만든 재사용 가능한 진단 알고리즘 · 2026-10-29", 12, color=NAVY, after=2)
para("Codeforces 실제 제출 기록 17,607,999건 · 코드: 바탕화면/캡스톤 2학기/capstone", 9.5, color=MUTED, after=8)

h1("1. 한눈에 보는 결과")
oj = D["전체 판정"].value_counts()
table(["항목", "결과"], [
    ["진단 대상", f"{len(D):,}명 (2024-01 ~ 06에 30문제 이상)"],
    ["전체 판정 (공식 레이팅 기준)", f"실력 부족 {oj.get('실력 부족', 0):,}명 / 레이팅 수준 {oj.get('레이팅 수준', 0):,}명 / 레이팅 이상 {oj.get('레이팅 이상', 0):,}명"],
    ["유형 약점·강점 (실효 레이팅 기준)", f"약점이 있는 사람 {(D['약점 카테고리'].notna()).sum():,}명, 강점이 있는 사람 {(D['강점 카테고리'].notna()).sum():,}명"],
    ["학습 유형", f"4개 · 안정성 {S['학습유형']['안정성_ARI']} · 레이팅 연관도 {S['학습유형']['레이팅_연관도']} (실력이 아니라 습관으로 나뉨)"],
    ["검증 대상", "진단 대상 중 2024-07 ~ 12에 10문제 이상 푼 8,345명"],
    ["검증 ② 진단의 쓸모", f"C 성공 예측 AUC {v3.iloc[0]['AUC']:.3f} → {v3.iloc[1]['AUC']:.3f}, log-loss {v3.iloc[0]['log-loss']:.4f} → {v3.iloc[1]['log-loss']:.4f} (진단을 더하면 예측이 조금 좋아짐)"],
    ["검증 ③ 실력 부족", f"실효 레이팅 AUC {v4.iloc[0]['AUC']:.3f} vs 공식 레이팅 {v4.iloc[1]['AUC']:.3f} (실효 레이팅이 더 정확)"],
    ["검증 ④ 학습 유형", "C 행동으로 다시 배정해도 같은 유형 유지 52.9% (유형별 인원 비율을 반영한 무작위 기준 32.9%, 카파 0.297). 유형별 차이 큼: 로직 점검형 64.1% ~ 문법 실수형 22.9%"],
    ["검증 ① 유형 약점", f"C 1문제당 (실제−기대): 약점 {v1.loc['약점','C_1문제당_실제_기대']*100:+.2f}%p < 보통 {v1.loc['보통','C_1문제당_실제_기대']*100:+.2f}%p < 강점 {v1.loc['강점','C_1문제당_실제_기대']*100:+.2f}%p — 보통·강점보다 상대적으로 낮은 성과(차이 −1.29%p, 95% −1.98 ~ −0.60). 단 약점도 기대보다 낮지는 않았고(+0.13%p), 단순 정답률로 고른 것보다 적중률이 조금 낮음(42.8% vs 43.0%)"],
], [5, 12.4])

h1("2. 팀 결정 기록 (설계 근거)")
table(["단계", "기준", "결정", "이유"], [
    ["0", "맞힌 뒤 오답", "첫 정답에서 자르기", "이미 푼 문제를 다른 방법으로 실험한 것이지 못 푼 게 아님"],
    ["0", "레이팅", "그 문제 첫 제출 때 레이팅", "문제를 풀기 시작할 때의 실력"],
    ["0", "PARTIAL(부분 점수)", "실패(기타 오답)", "완전한 정답이 아님. Codeforces도 OK만 푼 문제로 인정"],
    ["0", "채점 시스템 오류(REJECTED 등)", "제외", "풀이가 아니라 채점 쪽 문제"],
    ["1", "기간", "A 학습 2023-07~12 / B 진단 2024-01~06 / C 검증 2024-07~12", "검증 기간은 학습·진단에 한 번도 안 씀"],
    ["1", "기간 배정", "첫 제출한 날이 속한 기간", "헷갈리지 않고 미래 정보를 안 봄"],
    ["1", "관찰 부족", "기간 끝 7일 안에 시작한 문제 전부 제외", "못 푼 것만 빼면 성공률이 부풀려짐"],
    ["1", "진단 기간", "6개월 (대상 8,504명 vs 1년 9,978명, 85%)", "간단하게. 절반 이하로 줄지 않음"],
    ["2", "기대 확률", "A 기간으로 맞춘 곡선 (Elo 공식 그대로는 안 씀)", "Elo 그대로는 실제 93% 구간을 36%로 예측. A 6·12개월 차이 없음(log-loss 0.1932 vs 0.1930)"],
    ["3", "판정", "z ≤ −2 약점, z ≥ 2 강점, 카테고리 5문제 미만 판정 보류", "우연으로 볼 확률 약 5% 이하 (통계 관례)"],
    ["3", "실력 부족 vs 유형 약점", "공식 레이팅 기준 전체 z ≤ −2 = 실력 부족, 실효 레이팅 기준 카테고리 z ≤ −2 = 유형 약점", "본인 평소 실력 대비 특히 약한 유형만 골라냄 (설계 문서)"],
    ["4", "행동 지표", "설계 문서 6종 (오류 비율은 WA·TLE·MLE·RE·CE 4개 값)", "팀 문서에 계산식이 있어 설명하기 쉬움"],
    ["4", "표준화·군집", "레이팅 구간(1200·1400·1600·1900·2100) 안 표준화 → K-means 4개", "군집이 실력 순으로 갈리지 않게"],
], [1, 3.2, 6.6, 6.6], size=8.5)

h1("3. 단계별 결과와 그림")
h2("2단계 기대 확률 곡선")
para(f"확률 = 1 / (1 + exp(−({S['기대확률_모델']['a']:.4f} {S['기대확률_모델']['b']:+.4f} × (문제 난이도 − 레이팅)/400))), 5%~95%로 자름. A 기간 908,985개로 학습.")
image("그림1_기대확률_보정.png", 14)
para("한계: 자기 레이팅보다 800 이상 어려운 문제(시도의 약 2.7%)는 실제 84%인데 예측 73%로 낮게 나옴.", 9.5, color=MUTED)
h2("3단계 강점·약점 진단")
image("그림2_진단결과_분포.png")
bullet("실효 레이팅을 계산할 수 없는 사람(진단 기간에 모두 맞힘) 557명은 실효 레이팅을 '계산 불가'로 표시하고 카테고리 판정은 모두 보류.")
bullet("카테고리 판정: 보통 64,312 / 판정 보류 16,242 / 강점 1,937 / 약점 1,247 (사람×카테고리). 실효 레이팅 계산 불가 557명은 카테고리를 모두 판정 보류로 처리.")
h2("실제 사용자 사례 (규칙으로 선택)")
para("선택 규칙: B에서 80문제 이상, 기준일 레이팅 1000~1600, 약점 1개·강점 1~2개, 실효 레이팅 계산 가능 → 20명 중 문제 수가 가장 많은 사람(사례 1)과, '실력 부족'까지 함께 보이는 사람(사례 2).", 9.5, color=MUTED)
image("그림3_사례1_user12313.png", 15)
image("그림3_사례2_user1610.png", 15)
h2("4단계 학습 유형")
image("그림4_학습유형_프로필.png")
table(["유형", "인원", "두드러진 지표 (|중심| ≥ 0.5)", "이름 (팀 확정)"], [
    ["유형1", "1,606", S["학습유형"]["이름_근거"]["0"], "효율 개선 필요형"],
    ["유형2", "3,788", S["학습유형"]["이름_근거"]["1"], "로직 점검형"],
    ["유형3", "706", S["학습유형"]["이름_근거"]["2"], "문법 실수·빠른 재제출형"],
    ["유형4", "2,402", S["학습유형"]["이름_근거"]["3"], "도전적 시행착오형"],
], [1.6, 1.6, 9.2, 5])
para("유형 설명: " + " / ".join(f"{S['학습유형']['유형_이름'][k]} = {S['학습유형']['유형_설명'][k]}" for k in ["0","1","2","3"]), 9.5)
para("이름은 설계 문서 규칙대로 '평균보다 0.5 표준편차 이상 차이 나는 지표'만 보고 붙였고, 팀이 그대로 확정함(2026-10-29).", 9.5, color=MUTED)

h1("4. 검증 (C 기간, 설계 문서 ⑥)")
image("그림5_검증결과.png")
table(["검증", "통과 기준 (설계 문서)", "결과", "판단"], [
    ["① 유형 약점", "약점 태그의 (실제−기대)가 보통보다 뚜렷하게 낮음",
     f"약점 {v1.loc['약점','C_1문제당_실제_기대']*100:+.2f}%p vs 보통 {v1.loc['보통','C_1문제당_실제_기대']*100:+.2f}%p vs 강점 {v1.loc['강점','C_1문제당_실제_기대']*100:+.2f}%p (약점−보통 −1.29%p, 95% −1.98 ~ −0.60). C에서도 기대 이하: 약점 42.8%, 보통 33.5%, 강점 10.8%",
     "보통·강점보다 상대적으로 낮은 성과. 단 약점도 기대보다 낮지는 않음(+0.13%p)"],
    ["① 보조", "단순 정답률보다 잘 맞는가 (구현 가이드)", "같은 개수로 골랐을 때 적중률: 우리 42.8% / 단순 정답률 43.0% / 무작위 35.1%", "단순 정답률보다 조금 낮음"],
    ["② 진단의 쓸모", "(B) 레이팅+진단의 AUC가 (A) 레이팅만보다 높음", f"AUC {v3.iloc[0]['AUC']:.3f} → {v3.iloc[1]['AUC']:.3f}, log-loss {v3.iloc[0]['log-loss']:.4f} → {v3.iloc[1]['log-loss']:.4f} (AUC는 동점 처리하는 표준 함수 roc_auc_score로 계산)", "기준 충족 (향상 폭은 작음)"],
    ["③ 실력 부족", "실효 레이팅이 공식 레이팅보다 C 성공을 잘 맞힘", f"AUC {v4.iloc[0]['AUC']:.3f} vs {v4.iloc[1]['AUC']:.3f}, log-loss {v4.iloc[0]['log-loss']:.4f} vs {v4.iloc[1]['log-loss']:.4f}", "기준 충족"],
    ["④ 학습 유형", "같은 유형 유지 비율이 무작위보다 확실히 높음", "52.9% vs 무작위 32.9% (유형별 인원 비율 반영, 카파 0.297). 유형별: 로직 점검형 64.1%(무작위 47.2%), 도전적 시행착오형 57.8%(27.6%), 효율 개선 필요형 32.3%(18.8%), 문법 실수형 22.9%(6.3%)", "기준 충족 (유형별 차이 큼)"],
], [2.4, 4.4, 7.6, 3], size=8.5)

h1("5. 발표에서 솔직하게 말할 한계")
bullet("유형 약점은 하반기에 보통·강점보다 상대적으로 낮은 성과를 보였지만, 기대보다 낮지는 않았고(+0.13%p), 같은 개수를 '단순 정답률 낮은 순'으로 고른 것보다 적중률이 조금 낮았다(42.8% vs 43.0%).")
bullet("진단 결과를 더하면 하반기 예측이 좋아지지만 향상 폭은 작다(AUC 0.723 → 0.732). 실효 레이팅은 공식 레이팅보다 정확했다(0.743 vs 0.722).")
bullet("학습 유형 유지율은 유형별 차이가 크다(로직 점검형 64.1% ~ 문법 실수형 22.9%). 모든 유형이 유형별 인원 비율을 반영한 무작위 기준보다는 높다.")
bullet("자기 레이팅보다 800 이상 어려운 문제는 기대 확률이 실제보다 낮게 나온다(시도의 2.7%).")
bullet("진단 기간에 모두 맞힌 557명은 실효 레이팅을 계산할 수 없어 카테고리 판정을 보류한다.")
bullet("코드가 없어서 틀린 원인은 알 수 없다 → LLM 단계에서 사용자가 코드·질문을 주면 보완한다.")

h1("6. 재사용 방법과 파일")
table(["순서", "파일", "하는 일"], [
    ["0", "step0_prepare.py", "데이터 정리 (태그·카테고리, 행 정리, 사람×문제)"],
    ["1·2", "step1_periods_model.py", "3개 기간 나누기 + 기대 확률 곡선 학습·저장"],
    ["3", "step3_diagnosis.py", "실효 레이팅, 전체 판정, 카테고리 약점·강점"],
    ["4", "step4_learning_type.py", "행동 지표 6종 → 구간별 표준화 → K-means 4개, 기준값 저장"],
    ["5", "step5_new_user.py", "저장한 기준값으로 새 사용자 진단 (데이터셋 아이디 또는 --cf 실제 Codeforces 아이디)"],
    ["검증", "step6_validate.py", "C 기간 검증 ①~④"],
    ["결과", "step7_results.py", "최종 결과표, 그림, 결과요약.xlsx"],
    ["공통", "engine.py / model/model_settings.json", "재사용 함수 모음 / 모든 기준값을 저장한 파일"],
], [1.4, 5.4, 10.6])
bullet("재사용 확인: 300명을 '새 사용자 진단'으로 다시 계산 → 전체 계산 결과와 300명 모두 일치. 실제 Codeforces 아이디도 진단됨(최근 6개월).")
bullet("실행: 맥은 python3, 패키지는 pip install pandas numpy scikit-learn matplotlib openpyxl")

doc.save(out / "진단알고리즘_분석결과_요약.docx")
print("saved", out / "진단알고리즘_분석결과_요약.docx")
