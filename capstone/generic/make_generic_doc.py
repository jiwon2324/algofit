"""재사용 2단계(새 데이터셋 적용) 설명·검증 문서 만들기 → generic/output/재사용알고리즘_새데이터셋_적용.docx"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

G = Path(__file__).resolve().parent
O = G / "output"
ROOT = G.parent
plt.rcParams.update({"font.family": ["AppleGothic", "Malgun Gothic", "NanumGothic", "sans-serif"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})

T1 = pd.read_csv(O / "검증_비교표_최소도전학생수.csv")
T2 = pd.read_csv(O / "검증_비교표_실력구간수.csv")
T3 = pd.read_csv(O / "검증_비교표_판정선.csv")
X = pd.read_csv(O / "검증_분야판정_교차표.csv", index_col=0)
V1 = pd.read_csv(ROOT / "output" / "step6_①판정별.csv", index_col=0)
dec = json.loads((G / "범용_기준.json").read_text(encoding="utf-8"))
boj = (O / "백준형식_예시" / "보고서.txt").read_text(encoding="utf-8").splitlines()
AT = json.loads((O / "atcoder" / "요약.json").read_text(encoding="utf-8"))
at_rep = (O / "atcoder" / "검증_보고서.txt").read_text(encoding="utf-8").splitlines()
at_diff = pd.read_csv(O / "atcoder" / "검증①_난이도.csv")
main = T1[T1["문제당 최소 도전 학생 수"] == 5].iloc[0]
thr = T3[T3["판정선 |z|"] == 2.0].iloc[0]
typ = T2[T2["실력 구간 수"] == 5].iloc[0]

# 그림: 약점으로 판정된 분야가 다음 기간(C)에도 기대 이하인 비율
fig, ax = plt.subplots(figsize=(7, 3.6))
labels = ["원래 방법\n(공식 난이도·레이팅)", "범용 방법\n(풀이 기록만으로 추정)"]
weak = [V1.loc["약점", "C에서도_기대_이하"] * 100, thr["C에서도 기대 이하: 약점"] * 100]
norm = [V1.loc["보통", "C에서도_기대_이하"] * 100, thr["C에서도 기대 이하: 보통"] * 100]
x = range(2)
ax.bar([i - 0.18 for i in x], weak, 0.36, color="#E07A4F", label="B에서 약점 판정")
ax.bar([i + 0.18 for i in x], norm, 0.36, color="#B9BFC9", label="B에서 보통 판정")
for i in x:
    ax.text(i - 0.18, weak[i] + 0.8, f"{weak[i]:.0f}%", ha="center", fontsize=9)
    ax.text(i + 0.18, norm[i] + 0.8, f"{norm[i]:.0f}%", ha="center", fontsize=9)
ax.set_xticks(list(x), labels); ax.set_ylabel("C 기간에도 기대 이하인 비율 (%)"); ax.set_ylim(0, 55)
ax.set_title("약점 판정이 다음 기간에도 이어지는가"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(O / "그림_범용방법_C검증.png"); plt.close(fig)

doc = Document()
st = doc.styles["Normal"]; st.font.name = "맑은 고딕"; st.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕"); st.font.size = Pt(10)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2); s.top_margin = s.bottom_margin = Cm(1.8)


def table(rows, header):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
    for c, v in zip(t.rows[0].cells, header):
        c.text = str(v)
    for r in rows:
        for c, v in zip(t.add_row().cells, r):
            c.text = str(v)


B = lambda t: doc.add_paragraph(t, style="List Bullet")
doc.add_heading("재사용 진단 알고리즘 — 새 데이터셋(다른 채점 사이트) 적용", 0)
doc.add_paragraph("교수님 피드백 '아예 다른 데이터셋에도 적용되는 알고리즘'을 반영해, 문제 난이도·학생 레이팅이 없는 채점 사이트(예: 우리 학교 채점 사이트)에도 "
                  "같은 진단 알고리즘을 적용할 수 있게 확장했다. 기존 진단 결과·검증은 그대로 두고, 범용 버전을 따로 만들어 비교했다.")

doc.add_heading("1. 재사용의 두 단계", 1)
table([["① 같은 사이트의 새 학생", "저장된 기준(model_settings.json, 사이트별 문제난이도.csv)으로 다시 학습하지 않고 바로 진단", "Codeforces 300/300명, 백준 형식 100/100명 일치"],
       ["② 새 사이트(데이터셋)", "대응표 파일 1개만 새로 쓰고, 풀이 기록으로 난이도·실력을 추정한 뒤 같은 절차로 진단", "실제 다른 플랫폼 AtCoder 실데이터로 검증 (5절) + 레이팅을 지운 Codeforces로 검증 (4절)"]],
      ["단계", "방법", "확인"])

doc.add_heading("2. 적용 방법 (새 사이트 3단계)", 1)
B("제출 기록 CSV 준비: 학생 · 문제 · 제출시각 · 결과 · 분야(알고리즘 분류) 5개 열 (난이도·레이팅은 없어도 됨)")
B("대응표 작성: generic/platforms/baekjoon_style.json을 복사해 열 이름, 결과 이름(예: '맞았습니다!!' → 정답), 분야 태그(예: '다이나믹 프로그래밍' → DP)만 바꾼다")
B("실행: python generic/run_generic.py --platform <대응표> --data <CSV> --out <결과 폴더> → 진단_사용자.csv, 진단_분야.csv, 학습유형.csv, 문제난이도.csv")
B("새 학생 추가 진단: 같은 명령에 --difficulty <결과 폴더>/문제난이도.csv 를 붙이면 다시 추정하지 않고 저장된 기준으로 진단")

doc.add_heading("3. 팀이 정한 기준", 1)
table([[k.replace("_", " "), v] for k, v in dec["팀_결정"].items()] + [["판정선 (원래 값 유지)", "|z| ≥ 2"], ["실력 구간 수 (기본값)", "5 (추정 실력 5분위)"]], ["항목", "결정"])
doc.add_paragraph("Rasch 모형(1모수 문항반응이론): P(맞힘) = 1 / (1 + exp(−(실력 − 난이도))). 실력과 난이도를 같은 눈금 위에서 풀이 기록만으로 동시에 추정한다. "
                  "구현은 2단계에서 저장한 기대 확률 곡선 P = 1/(1+exp(−(a + b·(난이도−실력)/400)))과 같은 눈금을 쓰도록 바꿔 쓴 것이며(1 logit ≈ 635 레이팅 점수), "
                  "학생 실력과 문제 난이도를 번갈아 갱신하는 결합 최대우도로 추정한다. 이후 실효 레이팅·분야별 z·약점/강점 판정·학습 유형 배정은 원래 엔진 함수를 그대로 쓴다.")

doc.add_heading("4. 검증 — 레이팅을 지운 Codeforces 데이터", 1)
doc.add_paragraph("Codeforces 데이터(B 기간, 8,502명)에서 문제 난이도와 학생 레이팅을 일부러 지우고 범용 방법으로만 진단한 뒤, "
                  "원래 진단(공식 값 사용, C 기간 검증 완료)과 비교했다. 결과가 비슷하면 난이도·레이팅이 없는 데이터에도 같은 알고리즘을 쓸 수 있다는 근거가 된다.")
table([["학생 실력: 원래 실효 레이팅과의 상관", f"{main['실력 상관(원래 실효 레이팅)']:.3f}"],
       ["분야 판정(약점·보통·강점) 일치율", f"{main['분야 판정 일치율']:.1%} (카파 {main['분야 판정 카파']:.2f})"],
       ["새 방법이 약점이라고 한 것 중 원래도 약점 (정밀도)", f"{main['새 약점 중 원래도 약점(정밀도)']:.1%}"],
       ["원래 약점 중 새 방법도 약점 (재현율)", f"{main['원래 약점 중 다시 약점(재현율)']:.1%}"],
       ["학습 유형 일치율", f"{typ['유형 일치율']:.1%} (카파 {typ['유형 카파']:.2f})"],
       ["추정 난이도와 공식 난이도의 상관 (도전 5명 이상 문제)", "0.54 (순위 상관 0.57)"]], ["항목", "결과"])
doc.add_paragraph("분야 판정 교차표 (행: 원래, 열: 범용 방법)")
table([[i] + [int(v) for v in r] for i, r in X.iterrows()], ["원래 \\ 범용"] + list(X.columns))
doc.add_heading("다음 기간(C)에도 약점이 이어지는가", 2)
doc.add_picture(str(O / "그림_범용방법_C검증.png"), width=Cm(14))
doc.add_paragraph(f"범용 방법으로 약점 판정된 분야는 C 기간에도 기대보다 못 푼 비율이 {thr['C에서도 기대 이하: 약점']:.0%}로 보통({thr['C에서도 기대 이하: 보통']:.0%})보다 높았다. "
                  f"원래 방법({V1.loc['약점', 'C에서도_기대_이하']:.0%} vs {V1.loc['보통', 'C에서도_기대_이하']:.0%})과 방향이 같고, 차이는 조금 작다.")

doc.add_heading("5. 실제 다른 플랫폼 적용 — AtCoder 실데이터", 1)
doc.add_paragraph(f"AtCoder(일본 프로그래밍 대회 사이트)의 실제 제출 기록을 AtCoder Problems 공개 API로 받아 적용했다. 학생 {AT['학생']}명, 가입 이후 전체 제출 {AT['제출']:,}개 (점수제 최적화 대회 제외). "
                  "결과 이름(AC·WA·TLE·MLE·RE·CE·OLE)·문제 번호 체계·사용자가 모두 Codeforces와 다른 진짜 다른 데이터셋이며, 바꾼 것은 대응표(platforms/atcoder.json) 하나뿐이다. "
                  "AtCoder에는 공식 알고리즘 분류(태그)가 없어 분야별 약점·강점은 판정하지 않고(태그가 있는 사이트에서는 자동으로 켜짐), 실력·난이도 추정과 학습 유형을 검증했다. "
                  "진단 기간 2024-01~08, 검증 기간 2024-09~12.")
table([["진단 대상 (30문제 이상)", f"{AT['진단 대상']}명"],
       ["① 우리가 추정한 문제 난이도 vs AtCoder Problems가 대회 결과로 따로 계산한 난이도", f"순위 상관 {AT['난이도 순위 상관(5명 이상)']}, 상관 {AT['난이도 상관(5명 이상)']} (도전 5명 이상 문제)"],
       ["② 다음 기간 성공 예측 (학생을 반으로 나눠 한쪽 기록으로 난이도 추정 → 다른 쪽 예측, 나누기 10번 반복)", f"학생 실력 + 문제 난이도 평균 AUC {AT['AUC(실력+난이도)']} > 문제 난이도만 {AT['AUC(난이도만)']} (10번 모두 향상) → 추정한 학생 실력이 예측에 도움"],
       ["③ 학습 유형이 다음 기간에도 유지", f"{AT['유형 유지']:.1%} (무작위 기준 {AT['유형 무작위 기준']:.1%}, 카파 {AT['유형 카파']})"]], ["검증", "결과"])
doc.add_paragraph("난이도 상관 (도전 학생 수별)")
table(at_diff.values.tolist(), list(at_diff.columns))
doc.add_paragraph("실행 기록")
for line in at_rep:
    B(line.strip())

doc.add_heading("검증 과정 — 데이터와 계산을 하나씩 확인", 2)
table([["데이터가 진짜인가", "무작위 학생 4명의 기록을 공개 API에서 다시 받아 비교 → 제출 1,219개 모두 일치. 문제 주소를 AtCoder 공식 사이트에서 열어 실제 대회 문제임을 확인"],
       ["팀 규칙을 지켰나", "'처음 제출이 기간 안인 문제만 새 문제' 규칙을 지키기 위해 학생들의 가입 이후 전체 제출 기록을 받아 사용 (처음 2024년만 받았던 것을 고침)"],
       ["계산이 맞나", "학생 1명(Arcsecond)을 엔진 없이 따로 짠 코드로 다시 계산 → 문제 수·해결 수·실효 레이팅 모두 똑같음"],
       ["Rasch 추정기가 맞나", "정답(난이도·실력)을 미리 정해 만든 가짜 데이터로 시험 → 난이도 복원 상관 0.87, 실력 복원 상관 0.78"],
       ["문서의 실행 방법이 되는가", "run_generic.py --platform atcoder 로 그대로 실행되어 같은 결과가 나옴"]], ["확인 항목", "방법과 결과"])

doc.add_heading("이 알고리즘을 쓸 수 있는 조건", 2)
table([["제출 단위 기록", "학생 · 문제 · 제출 시각 · 결과(정답/오답 종류)가 제출마다 있어야 함. 최종 점수만 있는 기록은 안 됨"],
       ["정답/오답 판정", "부분 점수·최적화(점수제) 문제는 제외 (AtCoder 최적화 대회 AHC도 제외함)"],
       ["데이터 양", "학생 30문제 이상, 문제마다 도전 학생 5명 이상. 한 반 20~30명 규모면 믿을 수 있는 문제가 줄어듦"],
       ["알고리즘 분류(태그)", "분야별 약점·강점은 태그가 있어야 판정 (AtCoder처럼 없으면 실력·학습 유형만)"],
       ["전체 기록", "'첫 제출' 규칙을 지키려면 진단 기간 이전 기록도 필요"],
       ["거의 다 맞힌 학생", "실력을 계산할 수 없음 (AtCoder 진단 대상의 약 18%) — 원래 방법과 같은 규칙으로 판정 보류"],
       ["기대 확률 곡선", "Codeforces로 학습한 곡선을 그대로 씀. AtCoder에서는 예측 성공률이 실제보다 조금 낮았음 → 사이트별 곡선 재보정은 향후 과제"]],
      ["조건", "내용"])

doc.add_heading("6. 형식 변환 시연 (백준 형식)", 1)
doc.add_paragraph("한글 결과 이름·한글 알고리즘 분류를 쓰는 사이트(백준·학교 채점 사이트 등)도 대응표만으로 되는지 보이기 위해, Codeforces 제출 기록(학생 1,500명)의 "
                  "열 이름·결과 이름·알고리즘 분류·문제 번호·아이디를 백준 형식으로 바꾸고 난이도·레이팅을 지운 시험용 파일로 실행했다. ※ 실제 백준 데이터가 아니라 형식만 바꾼 시험용이다(실데이터 검증은 5절 AtCoder).")
for line in boj[1:]:
    B(line.strip())
B("새 학생 1명씩 저장된 기준(문제난이도.csv·구간 표준화)으로 진단 → 전체 실행 결과와 100/100명 일치")

doc.add_heading("7. 비교 실험 (팀 결정 근거)", 1)
doc.add_paragraph("문제별 최소 응답자(도전 학생) 수")
table(T1[["문제당 최소 도전 학생 수", "진단에 쓴 사람×문제", "실력 상관(원래 실효 레이팅)", "분야 판정 일치율", "분야 판정 카파", "원래 약점 중 다시 약점(재현율)", "새 약점 중 원래도 약점(정밀도)"]].values.tolist(),
      ["최소 응답자", "사용 사람×문제", "실력 상관", "판정 일치율", "카파", "재현율", "정밀도"])
doc.add_paragraph("판정선 (참고: 원래 값 2.0 유지)")
table(T3[["판정선 |z|", "새 약점 수", "분야 판정 일치율", "원래 약점 재현율", "새 약점 정밀도", "C에서도 기대 이하: 약점", "C에서도 기대 이하: 보통"]].values.tolist(),
      ["판정선", "약점 수", "일치율", "재현율", "정밀도", "C 기대 이하(약점)", "C 기대 이하(보통)"])
doc.add_paragraph("실력 구간 수 (학습 유형 표준화)")
table(T2.values.tolist(), list(T2.columns))

doc.add_heading("8. 한계", 1)
B("공식 난이도 없이 같은 데이터로 난이도를 추정하면 z가 조금 작게 나와, 원래 약점 중 42%만 다시 약점으로 잡힌다(대신 잡은 것은 92%가 맞음). 약점을 놓치는 쪽으로 보수적이다.")
B("다음 기간 예측력은 원래 방법과 방향은 같지만 차이가 약 절반 수준이다. 공식 난이도가 있으면 넣어서 쓰는 것이 더 정확하다(대응표에 열 이름만 적으면 사용).")
B("문제별 응답자가 적은 작은 학교 데이터에서는 난이도를 믿을 수 있는 문제가 줄어든다(시연: 1,500명 기준 7,925문제 중 4,596문제 사용).")
B("공식 레이팅이 없으므로 '전체 판정(레이팅 대비 실력 부족/이상)'은 제공하지 않는다(팀 결정).")
B("AtCoder는 공식 알고리즘 분류가 없어 실데이터에서 분야별 판정은 검증하지 못했다. 분류가 있는 실제 학교·백준 데이터 적용은 향후 과제다(백준 형식은 형식 변환 시연만).")
out = O / "재사용알고리즘_새데이터셋_적용.docx"
doc.save(out)
print("저장:", out)
