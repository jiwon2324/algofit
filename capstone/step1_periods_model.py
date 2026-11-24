"""
1단계. 기간 나누기  +  2단계. 기대 확률 모델

  1단계: 0단계에서 정리한 제출(output/submissions_clean.csv)을 3개 기간으로 나눠 사람×문제 표를 만든다.
         A 모델 학습 / B 진단 / C 검증  (문제는 '처음 제출한 날'이 속한 기간에 넣는다)
  2단계: A 기간으로 '레이팅 차이 → 성공 확률' 곡선을 학습하고, 계수를 파일로 저장한다.
         (Elo 공식을 그대로 쓰지 않고, 같은 모양의 곡선을 실제 데이터로 다시 맞춘 것)

실행: 먼저 step0_prepare.py를 실행한 뒤, 터미널에서  python step1_periods_model.py
결과: output 폴더에 기간별 사람×문제 표, model 폴더에 저장한 기준값(model_settings.json)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# [설정] 팀이 정한 값 (2026-10-25 확정)
# ============================================================
BASE_DATE = "2024-07-01"     # 기준일 = B(진단) 끝 = C(검증) 시작
DIAG_MONTHS = 6              # B 진단 기간 길이
TRAIN_MONTHS = 6             # A 모델 학습 기간 길이 (B 바로 앞)
SHORT_OBS_DAYS = 7           # 각 기간 끝 7일 안에 시작한 문제는 전부 제외
MIN_USER_PROBLEMS = 30       # 진단 대상 최소 문제 수 (B 기간)
MIN_VALID_PROBLEMS = 10      # 검증 대상 최소 문제 수 (C 기간)
P_CLIP = (0.05, 0.95)        # 기대 확률을 이 범위로 자름 (z 계산 분모가 0이 되지 않게)
OUTPUT_DIR = "output"
MODEL_DIR = "model"
# ============================================================

out, mdir = Path(OUTPUT_DIR), Path(MODEL_DIR)
mdir.mkdir(exist_ok=True)
report = []


def log(msg):
    print(msg)
    report.append(msg)


# 0단계의 '사람×문제 묶기' 함수를 그대로 가져와서 쓴다 (규칙을 한 곳에서만 관리)
src = Path("step0_prepare.py").read_text(encoding="utf-8")
G = {"pd": pd}
exec(src[src.index("ERR_COLS = {"):src.index("up = make_user_problem(df)")], G)
make_user_problem = G["make_user_problem"]

clean = out / "submissions_clean.csv"
if not clean.exists():
    raise SystemExit("[오류] output/submissions_clean.csv가 없습니다. step0_prepare.py를 먼저 실행하세요.")
s = pd.read_csv(clean, parse_dates=["time"])

# ------------------------------------------------------------
# 1단계. 기간 나누기
# ------------------------------------------------------------
B_end = pd.Timestamp(BASE_DATE)
B_start = B_end - pd.DateOffset(months=DIAG_MONTHS)
A_start = B_start - pd.DateOffset(months=TRAIN_MONTHS)
C_end = s["time"].max().floor("D") + pd.Timedelta(days=1)          # 데이터 마지막 날의 다음 날

periods = {}
for name, st, en in [("A", A_start, B_start), ("B", B_start, B_end), ("C", B_end, C_end)]:
    t = make_user_problem(s, st, en, SHORT_OBS_DAYS)
    t = t[t["n_judged"] > 0]                                         # 컴파일 에러만 있는 문제 제외
    periods[name] = t
    log(f"[1단계] {name} {st.date()} ~ {en.date()}: 사람×문제 {len(t):,}줄, 사용자 {t['handle'].nunique():,}명, 성공 비율 {t['solved'].mean():.1%}")

cnt_b = periods["B"].groupby("handle").size()
diag_users = set(cnt_b[cnt_b >= MIN_USER_PROBLEMS].index)
cnt_c = periods["C"][periods["C"]["handle"].isin(diag_users)].groupby("handle").size()
valid_users = set(cnt_c[cnt_c >= MIN_VALID_PROBLEMS].index)
log(f"[1단계] 진단 대상 (B {MIN_USER_PROBLEMS}문제 이상): {len(diag_users):,}명")
log(f"[1단계] 검증 대상 (진단 대상 중 C {MIN_VALID_PROBLEMS}문제 이상): {len(valid_users):,}명")

periods["A"].to_csv(out / "user_problem_A_train.csv", index=False, encoding="utf-8-sig")
periods["B"][periods["B"]["handle"].isin(diag_users)].to_csv(out / "user_problem_B_diag.csv", index=False, encoding="utf-8-sig")
periods["C"][periods["C"]["handle"].isin(valid_users)].to_csv(out / "user_problem_C_valid.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"handle": sorted(diag_users)}).to_csv(out / "users_diag.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# 2단계. 기대 확률 모델 (A 기간으로 학습)
#   확률 = 1 / (1 + exp(-(a + b × 차이)))     차이 = (문제 난이도 − 첫 제출 레이팅) / 400
# ------------------------------------------------------------
def fit_logistic(x, y, iters=30):
    """입력 1개짜리 로지스틱 회귀 (뉴턴 방법). 외부 라이브러리 없이 pandas·numpy만 사용."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        H = (X * (p * (1 - p))[:, None]).T @ X
        w = w + np.linalg.solve(H, X.T @ (y - p))
    return w


def expect(diff, w, clip=True):
    p = 1 / (1 + np.exp(-(w[0] + w[1] * diff)))
    return np.clip(p, *P_CLIP) if clip else p


A = periods["A"]
xA = ((A["problem_rating"] - A["first_rating"]) / 400).to_numpy(float)
w = fit_logistic(xA, A["solved"].to_numpy(float))
log(f"[2단계] A 기간 {len(A):,}개로 학습: 확률 = 1 / (1 + exp(-({w[0]:.4f} {w[1]:+.4f} × 차이/400)))")


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = np.mean((p - y) ** 2)
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    auc = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
    return ll, brier, auc


B = periods["B"][periods["B"]["handle"].isin(diag_users)]
xB = ((B["problem_rating"] - B["first_rating"]) / 400).to_numpy(float)
yB = B["solved"].to_numpy(float)
for nm, p in [("Elo 공식 그대로", 1 / (1 + 10 ** xB)), ("학습한 곡선", expect(xB, w, clip=False))]:
    ll, br, auc = metrics(yB, p)
    log(f"[2단계] B 기간 확인 · {nm}: log-loss {ll:.4f}, Brier {br:.4f}, AUC {auc:.4f}")

bins = [-np.inf, -800, -400, -200, 0, 200, 400, 800, np.inf]
lab = ["~-800", "-800~-400", "-400~-200", "-200~0", "0~200", "200~400", "400~800", "800~"]
cal = pd.DataFrame({"구간": pd.cut(B["problem_rating"] - B["first_rating"], bins, labels=lab),
                    "실제": yB, "예측": expect(xB, w, clip=False)}).groupby("구간", observed=True).agg(
    시도=("실제", "size"), 실제_성공비율=("실제", "mean"), 예측_평균=("예측", "mean"))
cal.to_csv(out / "step2_calibration_B.csv", encoding="utf-8-sig")
log("[2단계] B 기간 보정표 (난이도 − 레이팅 구간별):\n" + cal.round(3).to_string())

# ------------------------------------------------------------
# 저장: 새 데이터에 그대로 적용할 기준값 (설정 파일 하나)
# ------------------------------------------------------------
settings = {
    "결정일": "2026-10-25",
    "기간": {"기준일": BASE_DATE, "진단_개월": DIAG_MONTHS, "학습_개월": TRAIN_MONTHS,
             "관찰부족_제외_일수": SHORT_OBS_DAYS, "A": [str(A_start.date()), str(B_start.date())],
             "B": [str(B_start.date()), str(B_end.date())], "C": [str(B_end.date()), str(C_end.date())]},
    "최소_문제수": {"진단_대상": MIN_USER_PROBLEMS, "검증_대상": MIN_VALID_PROBLEMS},
    "기대확률_모델": {"식": "1 / (1 + exp(-(a + b * (문제난이도 - 첫제출레이팅) / 400)))",
                    "a": float(w[0]), "b": float(w[1]), "자르기": list(P_CLIP),
                    "학습_데이터": f"A 기간 사람×문제 {len(A):,}개"},
    "verdict_분류": {"성공": G.get("SUCCESS", ["OK"]), "참고": "실패·행동지표 전용·제외 목록은 step0_prepare.py 설정과 같음"},
    "한계": ["레이팅보다 800 이상 어려운 문제는 실제보다 낮게 예측됨 (B 기간 시도의 약 2.7%)"],
    "아직_팀이_정할_것": ["3단계 판정 기준 (z, 카테고리 최소 문제 수)", "실력 부족 vs 유형 약점 정의", "4단계 행동 지표·표준화·군집"],
}
(mdir / "model_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"[저장] {mdir / 'model_settings.json'}")

(out / "step1_2_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\n끝! output/step1_2_report.txt 와 model/model_settings.json 을 확인하세요.")
