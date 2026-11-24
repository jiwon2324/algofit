"""
4단계. 학습 유형 분류 (B 진단 기간)
  ① 사람마다 행동 지표 6종 계산 (설계 문서 4-1)
  ② 로그 변환 → 레이팅 구간별 표준화 (구간별 평균·표준편차를 저장)
  ③ K-means (k=4) → 군집 중심 저장, 안정성·레이팅 연관도 확인
  ④ 이름 붙이기 근거: 군집 중심이 평균보다 0.5 표준편차 이상 높거나 낮은 지표 (설계 문서 4-2)
실행: python step4_learning_type.py   (step3_diagnosis.py 실행 후)
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import engine

t0 = time.time()
out = Path("output")
S = engine.load_settings()
B0, B1 = pd.Timestamp(S["기간"]["B"][0]), pd.Timestamp(S["기간"]["B"][1])

up = pd.read_csv(out / "user_problem_B_diag.csv", parse_dates=["first_time", "solved_time"])
D = pd.read_csv(out / "step3_diagnosis_users.csv").set_index("handle")

# ① 재제출 간격: B 표에 있는 (사람, 문제)의 제출만, 첫 정답까지
sub = pd.read_csv(out / "submissions_clean.csv", usecols=["handle", "problem_id", "time", "verdict_group"], parse_dates=["time"])
sub = sub[(sub["time"] >= B0) & (sub["time"] < B1)]
sub = sub.merge(up[["handle", "problem_id", "solved_time"]], on=["handle", "problem_id"], how="inner")
sub = sub[sub["solved_time"].isna() | (sub["time"] <= sub["solved_time"])]
gaps = {h: engine.regap_minutes(g) for h, g in sub.groupby("handle")}

X = pd.DataFrame({h: engine.behavior_metrics(g, gaps.get(h, pd.Series(dtype=float))) for h, g in up.groupby("handle")}).T
X.index.name = "handle"
X["레이팅 구간"] = engine.rating_band(D.loc[X.index, "기준일 레이팅"].to_numpy())
n_nogap = int(X["재제출 간격(분)"].isna().sum())
X["재제출 간격(분)"] = X["재제출 간격(분)"].fillna(X.groupby("레이팅 구간")["재제출 간격(분)"].transform("median"))   # 틀린 적이 없는 사람은 같은 구간 중앙값

# ② 로그 변환 → 구간별 표준화 (평균·표준편차 저장)
T = engine.transform(X[engine.FEATURES].astype(float))
band_stats = {b: {"평균": T[X["레이팅 구간"] == b].mean().to_dict(), "표준편차": T[X["레이팅 구간"] == b].std().to_dict(),
                  "인원": int((X["레이팅 구간"] == b).sum())} for b in engine.RATING_BAND_NAMES}
S["학습유형"] = {"지표": engine.FEATURES, "로그변환": engine.LOG_FEATURES, "레이팅_구간_경계": engine.RATING_BANDS,
                 "레이팅_구간_이름": engine.RATING_BAND_NAMES, "구간별_표준화": band_stats}
Z = engine.standardize(T, X["레이팅 구간"].to_numpy(), S)

# ③ K-means
km = KMeans(n_clusters=engine.N_TYPES, n_init=20, random_state=42).fit(Z[engine.FEATURES])
lab = km.labels_
centers = pd.DataFrame(km.cluster_centers_, columns=engine.FEATURES)
sil = silhouette_score(Z[engine.FEATURES], lab, sample_size=6000, random_state=42)
aris = []
for s_ in range(5):
    idx = np.random.default_rng(s_).choice(len(Z), size=int(0.8 * len(Z)), replace=False)
    aris.append(adjusted_rand_score(lab, KMeans(n_clusters=engine.N_TYPES, n_init=10, random_state=s_).fit(Z.iloc[idx][engine.FEATURES]).predict(Z[engine.FEATURES])))


def cramers_v(a, b):
    ct = pd.crosstab(a, b).to_numpy(); n = ct.sum()
    e = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    return math.sqrt(((ct - e) ** 2 / e).sum() / (n * (min(ct.shape) - 1)))


cv = cramers_v(lab, X["레이팅 구간"])

# ④ 이름 붙이기 근거: |중심| ≥ 0.5 인 지표 (높음↑ / 낮음↓)
desc = {}
for k in range(engine.N_TYPES):
    c = centers.loc[k].sort_values(key=abs, ascending=False)
    desc[k] = ", ".join(f"{f}{'↑' if v > 0 else '↓'}({v:+.2f})" for f, v in c.items() if abs(v) >= engine.NAME_TH) or "두드러진 지표 없음"

S["학습유형"].update({"군집_중심": km.cluster_centers_.tolist(), "유형_이름": {str(k): f"유형{k + 1}" for k in range(engine.N_TYPES)},
                     "이름_근거": {str(k): desc[k] for k in range(engine.N_TYPES)}, "인원": {str(k): int((lab == k).sum()) for k in range(engine.N_TYPES)},
                     "실루엣": round(float(sil), 3), "안정성_ARI": round(float(np.mean(aris)), 3), "레이팅_연관도": round(cv, 3)})
S["아직_팀이_정할_것"] = ["유형 이름 4개 (이름_근거를 보고 결정)"]
engine.save_settings(S)

X["유형번호"] = lab + 1
X.to_csv(out / "step4_behavior_types.csv", encoding="utf-8-sig")
prof = X.groupby("유형번호")[engine.FEATURES].mean()
prof.insert(0, "인원", X["유형번호"].value_counts().sort_index())
prof.to_csv(out / "step4_type_profile.csv", encoding="utf-8-sig")

rep = [f"[4단계] 대상 {len(X):,}명, 지표 {len(engine.FEATURES)}개 (재제출 간격 없는 {n_nogap:,}명은 같은 구간 중앙값)",
       "[4단계] 레이팅 구간별 인원: " + ", ".join(f"{b} {band_stats[b]['인원']:,}" for b in engine.RATING_BAND_NAMES),
       f"[4단계] K-means k={engine.N_TYPES}: 실루엣 {sil:.3f}, 안정성(ARI 5회 평균) {np.mean(aris):.3f}, 레이팅 연관도(Cramér's V) {cv:.3f}",
       "[4단계] 유형별 실제 평균값:\n" + prof.round(3).to_string(),
       "[4단계] 군집 중심 (구간 안 표준점수):\n" + centers.round(2).rename(index=lambda k: f"유형{k + 1}").to_string(),
       "[4단계] 이름 붙이기 근거 (|중심| ≥ 0.5):\n" + "\n".join(f"  유형{k + 1} ({(lab == k).sum():,}명): {desc[k]}" for k in range(engine.N_TYPES)),
       f"({time.time() - t0:.0f}초)"]
(out / "step4_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
