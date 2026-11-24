"""
실제 다른 플랫폼(AtCoder) 데이터로 재사용 알고리즘 검증
  - 데이터: AtCoder Problems 공개 API로 받은 실제 제출 기록 (2024년, fetch_atcoder.py)
  - 진단 기간 B = 2024-01-01 ~ 2024-09-01, 검증 기간 C = 2024-09-01 ~ 2025-01-01 (Codeforces 검증과 같은 방식)
  - 바꾼 것은 대응표(platforms/atcoder.json) 하나뿐, 알고리즘 코드·기준값은 그대로
검증
  ① 우리가 풀이 기록만으로 추정한 문제 난이도 vs AtCoder Problems가 대회 결과로 따로 계산한 난이도 (서로 독립)
  ② B에서 추정한 학생 실력으로 C 기간 성공 여부를 맞히는가 (AUC) — 학생 실력을 빼면 얼마나 떨어지는가
  ③ 학습 유형이 C 기간에도 유지되는가 (무작위 기준·카파)
실행: cd capstone && .venv/bin/python generic/validate_atcoder.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import generic_engine as ge

t0 = time.time()
G = Path(__file__).resolve().parent
OUT = G / "output" / "atcoder"; OUT.mkdir(parents=True, exist_ok=True)
S = ge.engine.load_settings()
B0, B1, C1 = "2024-01-01", "2024-09-01", "2025-01-01"
rep = []


def log(m):
    print(m, flush=True); rep.append(m)


raw = pd.read_csv(G / "sample" / "atcoder_제출기록.csv")
cfg = ge.load_platform("atcoder")
sub, srep = ge.standardize(raw, cfg)
log(f"[데이터] AtCoder 실제 제출 {len(raw):,}개, 학생 {raw['user_id'].nunique()}명, 문제 {raw['problem_id'].nunique():,}개 (가입 이후 전체 기록, 최적화 대회 제외 — 진단·검증은 2024년)")
log(f"[1] 공통 형식 변환: {srep}")
log("    결과 종류: " + ", ".join(f"{k} {v:,}" for k, v in raw["result"].value_counts().items()))

# B 진단
up, cut = ge.make_user_problem(sub, B0, B1, S["기간"]["관찰부족_제외_일수"])
cnt = up.groupby("handle").size()
diag = cnt[cnt >= S["최소_문제수"]["진단_대상"]].index
diff, _ = ge.estimate_difficulty(up, S)
D, C, upd = ge.diagnose(up[up["handle"].isin(diag)], diff, S)
X, stats = ge.learning_types(upd, cut, D, S)
D["학습 유형"] = X["학습 유형"].reindex(D.index)
D.to_csv(OUT / "진단_사용자.csv", encoding="utf-8-sig"); X.to_csv(OUT / "학습유형.csv", encoding="utf-8-sig")
diff.to_csv(OUT / "추정_문제난이도.csv", index=False, encoding="utf-8-sig")
(OUT / "학습유형_구간별_표준화.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"[2] B 기간 사람×문제 {len(up):,}줄, 진단 대상 {len(D):,}명 (30문제 이상), 문제 {len(diff):,}개 중 도전 5명 이상 {int((diff['도전 학생 수'] >= 5).sum()):,}개")
log("    학습 유형: " + ", ".join(f"{k} {v}명" for k, v in X["학습 유형"].value_counts().items()))

# ① 난이도: 우리 추정 vs AtCoder Problems 난이도 (독립 계산)
pm = pd.read_csv(G / "sample" / "atcoder_문제난이도_검증용.csv").dropna(subset=["difficulty"])
pm = pm[pm["is_experimental"] != True]
M = diff.merge(pm, on="problem_id")
rows = []
for k in [5, 10, 20]:
    m = M[M["도전 학생 수"] >= k]
    rows.append({"도전 학생 최소": k, "문제 수": len(m), "상관": round(m[["추정 난이도", "difficulty"]].corr().iloc[0, 1], 3),
                 "순위 상관": round(m[["추정 난이도", "difficulty"]].corr("spearman").iloc[0, 1], 3)})
    log(f"[①] 난이도 (도전 {k}명 이상 문제 {len(m):,}개): 상관 {rows[-1]['상관']}, 순위 상관 {rows[-1]['순위 상관']}")
pd.DataFrame(rows).to_csv(OUT / "검증①_난이도.csv", index=False, encoding="utf-8-sig")

# ② C 기간 예측 — 공정하게: 학생을 둘로 나눠, 한쪽 학생들의 C 풀이 기록으로 문제 난이도를 추정하고 다른 쪽을 예측 (서로 바꿔 한 번 더)
upC_all, cutC = ge.make_user_problem(sub, B1, C1, 0)
eff = D.loc[D["실효 레이팅 비고"] == "", "실효 레이팅"]
targets = upC_all[upC_all["handle"].isin(eff.index)]
targets = targets[targets.groupby("handle")["problem_id"].transform("size") >= S["최소_문제수"]["검증_대상"]]
res = []
for seed in range(10):                                         # 학생 나누기를 10번 바꿔 가며 반복 (우연인지 확인)
    hs = np.array(sorted(targets["handle"].unique())); np.random.default_rng(seed).shuffle(hs)
    parts = []
    for f in [set(hs[::2]), set(hs[1::2])]:
        dC, _ = ge.estimate_difficulty(upC_all[~upC_all["handle"].isin(f)], S)   # 예측할 학생들의 기록은 빼고 난이도 추정
        t = targets[targets["handle"].isin(f)].merge(dC, on="problem_id")
        parts.append(t[t["도전 학생 수"] >= ge.MIN_PROBLEM_ATTEMPTS])
    E = pd.concat(parts)
    y = E["solved"].to_numpy()
    p_full = ge.engine.expect_prob(E["추정 난이도"], E["handle"].map(eff), S)
    p_noskill = ge.engine.expect_prob(E["추정 난이도"], np.full(len(E), eff.median()), S)
    res.append((roc_auc_score(y, p_full), roc_auc_score(y, p_noskill)))
res = np.array(res)
auc_full, auc_ns = res[:, 0].mean(), res[:, 1].mean()
log(f"[②] C 기간 성공 예측 — 학생을 반으로 나눠 교차 검증, 나누기 10번 반복 (학생 {E['handle'].nunique()}명, 시도 약 {len(E):,}개, 실제 성공 비율 {y.mean():.1%})")
log(f"    B에서 추정한 학생 실력 + 문제 난이도: 평균 AUC {auc_full:.3f}  /  학생 실력 없이 문제 난이도만: 평균 AUC {auc_ns:.3f}")
log(f"    10번 중 학생 실력이 예측을 높인 횟수: {int((res[:, 0] > res[:, 1]).sum())}번 (차이 {np.min(res[:, 0] - res[:, 1]):+.3f} ~ {np.max(res[:, 0] - res[:, 1]):+.3f})")
log(f"    예측 성공 비율 평균 {p_full.mean():.1%} vs 실제 {y.mean():.1%}")
upC = upC_all.merge(ge.estimate_difficulty(upC_all, S)[0], on="problem_id")
upC = upC[upC["도전 학생 수"] >= ge.MIN_PROBLEM_ATTEMPTS]
diffC = ge.estimate_difficulty(upC_all, S)[0]

# ③ 학습 유형 유지
DC, _, updC = ge.diagnose(upC.drop(columns=["추정 난이도", "도전 학생 수"])[lambda d: d["handle"].isin(E["handle"].unique())], diffC, S)
XC, _ = ge.learning_types(updC, cutC, DC, S, saved=stats)
both = XC.index.intersection(X.index)
bB, cC = X.loc[both, "유형번호"].to_numpy(), XC.loc[both, "유형번호"].to_numpy()
same = (bB == cC).mean()
pB, pC = pd.Series(bB).value_counts(normalize=True), pd.Series(cC).value_counts(normalize=True)
p_chance = float(sum(pB.get(t, 0) * pC.get(t, 0) for t in range(1, 5)))
log(f"[③] 학습 유형 C 기간 유지: {same:.1%} (학생 {len(both)}명), 무작위 기준 {p_chance:.1%}, 카파 {cohen_kappa_score(bB, cC):.3f}")

summary = {"제출": len(raw), "학생": int(raw["user_id"].nunique()), "진단 대상": len(D), "난이도 순위 상관(5명 이상)": rows[0]["순위 상관"],
           "난이도 상관(5명 이상)": rows[0]["상관"], "AUC(실력+난이도)": round(auc_full, 3), "AUC(난이도만)": round(auc_ns, 3),
           "유형 유지": round(same, 3), "유형 무작위 기준": round(p_chance, 3), "유형 카파": round(cohen_kappa_score(bB, cC), 3)}
(OUT / "요약.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"끝 ({time.time() - t0:.0f}초)")
(OUT / "검증_보고서.txt").write_text("\n".join(rep), encoding="utf-8")
