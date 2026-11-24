"""
재사용 2단계 검증: Codeforces 데이터에서 '문제 난이도'와 '학생 레이팅'을 일부러 지우고
범용 엔진(풀이 기록만으로 난이도·실력 추정)으로 진단 → 원래 진단(공식 값 사용, C 기간 검증 완료)과 얼마나 같은지 비교

같은 결과가 나오면: 난이도·레이팅이 없는 다른 채점 사이트(학교 등)에도 같은 알고리즘을 쓸 수 있다는 근거
실행: cd capstone && .venv/bin/python generic/validate_without_ratings.py
결과: generic/output/검증_*.csv, 검증_보고서.txt
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import generic_engine as ge

t0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"; OUT.mkdir(exist_ok=True)
S = ge.engine.load_settings()
B0, B1 = S["기간"]["B"]
rep = []


def log(m):
    print(m, flush=True); rep.append(m)


# 1) 원래 제출 기록 → 공통 형식 (난이도·레이팅 열은 일부러 읽지 않음)
cfg = ge.load_platform("codeforces")
raw = pd.read_csv(ROOT / "output" / "submissions_clean.csv", usecols=["handle", "problem_id", "verdict", "time", "categories"],
                  dtype={"handle": "category", "problem_id": "category", "verdict": "category", "categories": "category"})
sub, srep = ge.standardize(raw, cfg)
del raw
log(f"[1] 공통 형식 변환: {srep} ({time.time() - t0:.0f}초)")

# 2) 사람×문제 표 (B 진단 기간, 팀 규칙 그대로)
up, sub_cut = ge.make_user_problem(sub, B0, B1, S["기간"]["관찰부족_제외_일수"])
cnt = up.groupby("handle").size()
diag = cnt[cnt >= S["최소_문제수"]["진단_대상"]].index
log(f"[2] B 기간 사람×문제 {len(up):,}줄, 진단 대상 {len(diag):,}명 ({time.time() - t0:.0f}초)")

# 3) 난이도·실력 동시 추정 (레이팅 없이)
diff, theta = ge.estimate_difficulty(up, S)
diff.to_csv(OUT / "추정_문제난이도.csv", index=False, encoding="utf-8-sig")
orig_up = pd.read_csv(ROOT / "output" / "user_problem_B_diag.csv", usecols=["problem_id", "problem_rating"]).drop_duplicates("problem_id")
cmp_p = diff.merge(orig_up, on="problem_id")
log(f"[3] 추정 난이도 vs Codeforces 공식 난이도 (문제 {len(cmp_p):,}개)")
for k in [1, 5, 10, 20]:
    m = cmp_p["도전 학생 수"] >= k
    log(f"    도전 학생 {k}명 이상 {int(m.sum()):,}개: 상관 {cmp_p.loc[m, ['추정 난이도', 'problem_rating']].corr().iloc[0, 1]:.3f}, "
        f"순위 상관 {cmp_p.loc[m, ['추정 난이도', 'problem_rating']].corr('spearman').iloc[0, 1]:.3f}")

# 원래 진단 결과
OD = pd.read_csv(ROOT / "output" / "step3_diagnosis_users.csv").set_index("handle")
OC = pd.read_csv(ROOT / "output" / "step3_diagnosis_categories.csv")
OT = pd.read_csv(ROOT / "output" / "step4_behavior_types.csv").set_index("handle")["유형번호"]

# 4) 기준 후보별 진단 → 원래 진단과 비교 (팀 결정용 비교표)
rows = []
for k in [1, 5, 10, 20]:
    D, C, upd = ge.diagnose(up[up["handle"].isin(diag)], diff, S, min_attempts=k)
    D.index = D.index.astype(str)
    both = D.index.intersection(OD.index)
    ok = (D.loc[both, "실효 레이팅 비고"] == "") & OD.loc[both, "실효 레이팅 비고"].isna()
    r_eff = np.corrcoef(D.loc[both][ok]["실효 레이팅"], OD.loc[both][ok]["실효 레이팅"])[0, 1]
    r_off = np.corrcoef(D.loc[both][ok]["실효 레이팅"], OD.loc[both][ok]["기준일 레이팅"])[0, 1]
    M = C.merge(OC, on=["handle", "category"], suffixes=("_새", "_원래"))
    M = M[(M["judge_새"] != "판정 보류") & (M["judge_원래"] != "판정 보류")]
    agree = (M["judge_새"] == M["judge_원래"]).mean()
    kappa = cohen_kappa_score(M["judge_원래"], M["judge_새"])
    w_o, w_n = M["judge_원래"] == "약점", M["judge_새"] == "약점"
    rows.append({"문제당 최소 도전 학생 수": k, "진단 인원": len(D), "진단에 쓴 사람×문제": len(upd),
                 "실력 상관(원래 실효 레이팅)": round(r_eff, 3), "실력 상관(공식 레이팅)": round(r_off, 3),
                 "분야 판정 일치율": round(agree, 4), "분야 판정 카파": round(kappa, 3),
                 "원래 약점 중 다시 약점(재현율)": round((w_o & w_n).sum() / max(w_o.sum(), 1), 3),
                 "새 약점 중 원래도 약점(정밀도)": round((w_o & w_n).sum() / max(w_n.sum(), 1), 3)})
    if k == ge.MIN_PROBLEM_ATTEMPTS:
        D_main, C_main, up_main = D, C, upd
        pd.crosstab(M["judge_원래"], M["judge_새"], margins=True).to_csv(OUT / "검증_분야판정_교차표.csv", encoding="utf-8-sig")
    log(f"[4] 최소 도전 {k}명: {rows[-1]} ({time.time() - t0:.0f}초)")
T1 = pd.DataFrame(rows)
T1.to_csv(OUT / "검증_비교표_최소도전학생수.csv", index=False, encoding="utf-8-sig")

# 5) 학습 유형: 실력 구간 수 후보별 → 원래 유형과 비교
rows = []
for nb in [3, 5, 6]:
    X, stats = ge.learning_types(up_main, sub_cut, D_main, S, bands=nb)
    X.index = X.index.astype(str)
    both = X.index.intersection(OT.index)
    rows.append({"실력 구간 수": nb, "비교 인원": len(both),
                 "유형 일치율": round(float((X.loc[both, "유형번호"] == OT.loc[both]).mean()), 4),
                 "유형 카파": round(cohen_kappa_score(OT.loc[both], X.loc[both, "유형번호"]), 3)})
    if nb == ge.ABILITY_BANDS:
        X_main = X
    log(f"[5] 실력 구간 {nb}개: {rows[-1]} ({time.time() - t0:.0f}초)")
T2 = pd.DataFrame(rows)
T2.to_csv(OUT / "검증_비교표_실력구간수.csv", index=False, encoding="utf-8-sig")

# 6) 판정선 후보: 같은 데이터로 난이도를 추정하면 z가 조금 작아짐 → 판정선을 팀이 정하도록 비교
OC_cmp = OC[OC["judge"] != "판정 보류"]
rows = []
C0, C1 = S["기간"]["C"]
upC, _ = ge.make_user_problem(sub, C0, C1, 0)
diffC, _ = ge.estimate_difficulty(upC, S)                      # C 기간 문제 난이도도 C 기간 풀이 기록만으로 추정
upC = upC.merge(diffC, on="problem_id")
upC = upC[(upC["도전 학생 수"] >= ge.MIN_PROBLEM_ATTEMPTS) & upC["handle"].isin(D_main.index)]
upC = upC[upC.groupby("handle")["problem_id"].transform("size") >= S["최소_문제수"]["검증_대상"]]
eff = D_main.loc[D_main["실효 레이팅 비고"] == "", "실효 레이팅"]
upC = upC[upC["handle"].isin(eff.index)]
upC["p"] = ge.engine.expect_prob(upC["추정 난이도"], upC["handle"].map(eff), S)
LC = upC.assign(cat=upC["categories"].str.split(";")).explode("cat")
VC = LC.groupby(["handle", "cat"]).agg(n=("solved", "size"), o=("solved", "sum"), e=("p", "sum")).reset_index()
VC = VC[VC["n"] >= ge.engine.MIN_CAT_PROBLEMS].rename(columns={"cat": "category"})
VC["resid"], VC["below"] = (VC["o"] - VC["e"]) / VC["n"], (VC["o"] < VC["e"]).astype(int)
for th in [2.0, 1.8, 1.6]:
    ge.engine.Z_TH = th
    Cj = C_main.copy()
    Cj["judge"] = [ge.engine.judge(n, z) if j != "판정 보류" else j for n, z, j in zip(Cj["n"], Cj["z"], Cj["judge"])]
    M = Cj.merge(OC_cmp, on=["handle", "category"], suffixes=("_새", "_원래"))
    M = M[M["judge_새"] != "판정 보류"]
    w_o, w_n = M["judge_원래"] == "약점", M["judge_새"] == "약점"
    V = Cj.merge(VC, on=["handle", "category"])
    V = V[V["judge"] != "판정 보류"]
    g = V.groupby("judge")
    rows.append({"판정선 |z|": th, "새 약점 수": int((Cj["judge"] == "약점").sum()), "분야 판정 일치율": round((M["judge_새"] == M["judge_원래"]).mean(), 4),
                 "원래 약점 재현율": round((w_o & w_n).sum() / max(w_o.sum(), 1), 3), "새 약점 정밀도": round((w_o & w_n).sum() / max(w_n.sum(), 1), 3),
                 "C 검증 쌍(약점)": int(g.size().get("약점", 0)),
                 "C 1문제당 실제−기대: 약점": round(g["resid"].mean().get("약점", np.nan), 4),
                 "C 1문제당 실제−기대: 보통": round(g["resid"].mean().get("보통", np.nan), 4),
                 "C에서도 기대 이하: 약점": round(g["below"].mean().get("약점", np.nan), 3),
                 "C에서도 기대 이하: 보통": round(g["below"].mean().get("보통", np.nan), 3)})
    log(f"[6] 판정선 {th}: {rows[-1]}")
ge.engine.Z_TH = 2.0
T3 = pd.DataFrame(rows)
T3.to_csv(OUT / "검증_비교표_판정선.csv", index=False, encoding="utf-8-sig")
log("    (참고) 원래 방법의 C 검증: " + pd.read_csv(ROOT / "output" / "step6_①판정별.csv").round(4).to_string().replace("\n", " | "))

D_main.to_csv(OUT / "범용진단_사용자.csv", encoding="utf-8-sig")
C_main.to_csv(OUT / "범용진단_분야.csv", index=False, encoding="utf-8-sig")
X_main.to_csv(OUT / "범용진단_학습유형.csv", encoding="utf-8-sig")
log(f"끝 ({time.time() - t0:.0f}초)")
(OUT / "검증_보고서.txt").write_text("\n".join(rep), encoding="utf-8")
