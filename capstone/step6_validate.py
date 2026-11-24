"""
검증 (C 기간) — 설계 문서 ⑥
  ① 유형 약점: B에서 약점인 카테고리가 C에서도 기대보다 못 푸는가 (보통보다 뚜렷하게 낮은가)
     + 같은 개수를 '단순 정답률 낮은 순'·'무작위'로 골랐을 때와 적중률 비교 (구현 가이드)
  ② 진단의 쓸모: C 성공 예측 — (A) 레이팅만 vs (B) 레이팅 + 진단 결과, AUC 비교
  ③ 실력 부족: 실효 레이팅 vs 기준일(공식) 레이팅 중 어느 쪽이 C 성공을 더 잘 맞히나
  ④ 학습 유형: C 행동으로 지표를 다시 계산 → 저장한 중심에 배정 → 같은 유형 유지 비율 vs 무작위(1/k)
대상: 진단 대상 중 C 기간에 10문제 이상 푼 사람. C 기간 카테고리 비교는 C에서 5문제 이상(진단과 같은 기준)인 경우만.
실행: python step6_validate.py   (step3·4 실행 후)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import engine

t0 = time.time()
out = Path("output")
S = engine.load_settings()
C0, C1 = pd.Timestamp(S["기간"]["C"][0]), pd.Timestamp(S["기간"]["C"][1])
rng = np.random.default_rng(42)

D = pd.read_csv(out / "step3_diagnosis_users.csv").set_index("handle")
CB = pd.read_csv(out / "step3_diagnosis_categories.csv")
up_c = pd.read_csv(out / "user_problem_C_valid.csv", parse_dates=["solved_time"])
valid = up_c["handle"].unique()
rep = [f"[검증] 대상 {len(valid):,}명 (진단 대상 중 C 기간 {S['최소_문제수']['검증_대상']}문제 이상), C 기간 {C0.date()} ~ {C1.date()}, 시도 {len(up_c):,}개"]

# 실효 레이팅이 계산 불가(모두 맞힘·모두 틀림)인 사람은 ①③에서 제외
eff_ok = D["실효 레이팅 비고"].isna() | (D["실효 레이팅 비고"] == "")
r_eff = D["실효 레이팅"].where(eff_ok)
up_c["r_eff"] = up_c["handle"].map(r_eff)
up_c["r_base"] = up_c["handle"].map(D["기준일 레이팅"])

# ---------------- ① 유형 약점
uc = up_c[up_c["r_eff"].notna()].copy()
uc["p_eff"] = engine.expect_prob(uc["problem_rating"], uc["r_eff"], S)
L = uc.assign(cat=uc["categories"].str.split(";")).explode("cat")
VC = L.groupby(["handle", "cat"]).agg(n_c=("solved", "size"), o_c=("solved", "sum"), e_c=("p_eff", "sum")).reset_index()
VC = VC[VC["n_c"] >= engine.MIN_CAT_PROBLEMS]
P = CB.rename(columns={"category": "cat"}).merge(VC, on=["handle", "cat"])
P = P[P["judge"] != "판정 보류"]
P["resid"] = (P["o_c"] - P["e_c"]) / P["n_c"]
P["below"] = (P["o_c"] < P["e_c"]).astype(int)
t1 = P.groupby("judge").agg(쌍=("resid", "size"), C_1문제당_실제_기대=("resid", "mean"), C에서도_기대_이하=("below", "mean")).reindex(["약점", "보통", "강점"])
rep.append(f"[① 유형 약점] B 판정별 C 결과 (사람×카테고리, C에서 {engine.MIN_CAT_PROBLEMS}문제 이상):\n" + t1.round(4).to_string())
# 같은 개수 비교
P["k"] = P.groupby("handle")["judge"].transform(lambda s: (s == "약점").sum())
sel = P[P["k"] > 0].copy()
sel["rnd"] = rng.random(len(sel))
pick = {"우리 진단(약점)": sel[sel["judge"] == "약점"],
        "단순 정답률 낮은 순": sel[sel.groupby("handle")["solve_rate"].rank(method="first") <= sel["k"]],
        "무작위": sel[sel.groupby("handle")["rnd"].rank(method="first") <= sel["k"]]}
t2 = pd.DataFrame({nm: {"고른 쌍": len(s), "C에서도 기대 이하(적중률)": s["below"].mean(), "C 1문제당 (실제−기대)": s["resid"].mean()} for nm, s in pick.items()}).T
rep.append("[① 유형 약점] 같은 개수로 약점 고르기 비교:\n" + t2.round(4).to_string())


# ---------------- ② 진단의 쓸모 (AUC)
def auc(y, p):
    from sklearn.metrics import roc_auc_score     # 예측값이 같은(동점) 경우를 평균 순위로 처리하는 표준 함수
    return roc_auc_score(y, p)


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_offset(off, X, y, iters=25):
    X = np.column_stack([np.ones(len(y)), X]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(off + X @ b)))
        H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-3 * np.eye(len(b))
        b += np.linalg.solve(H, X.T @ (y - mu) - 1e-3 * b)
    return b


zc = CB[CB["judge"] != "판정 보류"].set_index(["handle", "category"])["z"]
A = up_c.copy()
A["p_off"] = engine.expect_prob(A["problem_rating"], A["first_rating"], S)
Lz = A.reset_index().assign(cat=A["categories"].str.split(";").to_numpy()).explode("cat")
Lz["z"] = [zc.get((h, c), 0.0) for h, c in zip(Lz["handle"], Lz["cat"])]
A["zmean"] = Lz.groupby("index")["z"].mean().reindex(A.index).fillna(0).to_numpy()
test = A["handle"].isin(set(rng.choice(valid, size=len(valid) // 2, replace=False))).to_numpy()
y = A["solved"].to_numpy(float)
off = np.log(A["p_off"] / (1 - A["p_off"])).to_numpy()
bA = fit_offset(off[~test], np.zeros((int((~test).sum()), 0)), y[~test])
bB = fit_offset(off[~test], A["zmean"].to_numpy()[~test, None], y[~test])
pA = 1 / (1 + np.exp(-(off[test] + bA[0])))
pB = 1 / (1 + np.exp(-(off[test] + bB[0] + bB[1] * A["zmean"].to_numpy()[test])))
t3 = pd.DataFrame({"(A) 레이팅만": {"AUC": auc(y[test], pA), "log-loss": logloss(y[test], pA)},
                   "(B) 레이팅 + 진단(카테고리 z)": {"AUC": auc(y[test], pB), "log-loss": logloss(y[test], pB)}}).T
rep.append(f"[② 진단의 쓸모] C 성공 예측 (사용자 절반으로 맞추고 나머지 절반으로 평가, 평가 시도 {int(test.sum()):,}개):\n" + t3.round(4).to_string())

# ---------------- ③ 실력 부족: 실효 vs 공식 레이팅
E = up_c[up_c["r_eff"].notna()]
ye = E["solved"].to_numpy(float)
p_e = engine.expect_prob(E["problem_rating"], E["r_eff"], S)
p_b = engine.expect_prob(E["problem_rating"], E["r_base"], S)
t4 = pd.DataFrame({"실효 레이팅(B에서 계산)": {"AUC": auc(ye, p_e), "log-loss": logloss(ye, p_e)},
                   "기준일 공식 레이팅": {"AUC": auc(ye, p_b), "log-loss": logloss(ye, p_b)}}).T
rep.append(f"[③ 실력 부족] C 성공을 더 잘 맞히는 레이팅 (시도 {len(E):,}개, 실효 레이팅 계산 불가 {int((~eff_ok).sum()):,}명 제외):\n" + t4.round(4).to_string())
for j in ["실력 부족", "레이팅 수준", "레이팅 이상"]:
    hs = D.index[(D["전체 판정"] == j) & eff_ok]
    m = E["handle"].isin(hs).to_numpy()
    rep.append(f"   B '{j}' 사람들의 C 시도: 실제 성공 {ye[m].mean():.3f} / 공식 레이팅 기대 {p_b[m].mean():.3f} / 실효 레이팅 기대 {p_e[m].mean():.3f}")

# ---------------- ④ 학습 유형 유지
X4 = pd.read_csv(out / "step4_behavior_types.csv").set_index("handle")
sub = pd.read_csv(out / "submissions_clean.csv", usecols=["handle", "problem_id", "time", "verdict_group"], parse_dates=["time"])
sub = sub[(sub["time"] >= C0) & (sub["time"] < C1)].merge(up_c[["handle", "problem_id", "solved_time"]], on=["handle", "problem_id"])
sub = sub[sub["solved_time"].isna() | (sub["time"] <= sub["solved_time"])]
gaps = {h: engine.regap_minutes(g) for h, g in sub.groupby("handle")}
XC = pd.DataFrame({h: engine.behavior_metrics(g, gaps.get(h, pd.Series(dtype=float))) for h, g in up_c.groupby("handle")}).T
band = engine.rating_band(D.loc[XC.index, "기준일 레이팅"].to_numpy())
med = {b: float(np.expm1(S["학습유형"]["구간별_표준화"][b]["평균"]["재제출 간격(분)"])) for b in engine.RATING_BAND_NAMES}
XC["재제출 간격(분)"] = XC["재제출 간격(분)"].fillna(pd.Series([med[b] for b in band], index=XC.index))
ZC = engine.standardize(engine.transform(XC[engine.FEATURES].astype(float)), band, S)
labC, _ = engine.assign_type(ZC, S)
same = (labC + 1 == X4.loc[XC.index, "유형번호"].to_numpy())
k = len(S["학습유형"]["군집_중심"])
keep = pd.crosstab(X4.loc[XC.index, "유형번호"].to_numpy(), labC + 1, rownames=["B 유형"], colnames=["C 유형"], normalize="index")
bB = X4.loc[XC.index, "유형번호"].to_numpy(); cC = labC + 1
pB = pd.Series(bB).value_counts(normalize=True); pC = pd.Series(cC).value_counts(normalize=True)
p_e = float(sum(pB.get(t, 0) * pC.get(t, 0) for t in range(1, k + 1)))          # 유형별 인원 비율을 반영한 무작위 기준
kappa = (same.mean() - p_e) / (1 - p_e)
rep.append(f"[④ 학습 유형] C 행동으로 다시 배정했을 때 같은 유형 유지: {same.mean():.1%}, 대상 {len(XC):,}명")
rep.append(f"   무작위 기준: 균등 1/k = {1/k:.0%}, 유형별 인원 비율 반영 = {p_e:.1%} → 카파 계수 {kappa:.3f}")
per = pd.DataFrame({"B 인원": pd.Series(bB).value_counts().sort_index(),
                    "유지율": pd.Series(same).groupby(bB).mean(),
                    "무작위 기준(C에서 그 유형 비율)": pd.Series({t: pC.get(t, 0) for t in range(1, k + 1)})})
per["유지율 − 무작위"] = per["유지율"] - per["무작위 기준(C에서 그 유형 비율)"]
per.index = [S["학습유형"]["유형_이름"][str(t - 1)] for t in per.index]
per.to_csv(out / "step6_④유형별유지.csv", encoding="utf-8-sig")
rep.append("   유형별 유지율과 무작위 기준:\n" + per.round(3).to_string())
rep.append("   유형별 유지 비율 (행 = B 유형, 열 = C 유형):\n" + keep.round(3).to_string())

rep.append(f"({time.time() - t0:.0f}초)")
(out / "step6_validation_report.txt").write_text("\n".join(rep), encoding="utf-8")
for nm, t in [("①판정별", t1), ("①같은개수", t2), ("②예측", t3), ("③레이팅", t4)]:
    t.to_csv(out / f"step6_{nm}.csv", encoding="utf-8-sig")
keep.to_csv(out / "step6_④유형유지.csv", encoding="utf-8-sig")
print("\n".join(rep))
