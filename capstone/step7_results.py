"""
결과 정리 — 최종 결과표(LLM 힌트 입력)와 PPT용 그림
실행: python step7_results.py   (step3·4·6 실행 후)
결과: output/최종_진단결과표.csv, output/결과요약.xlsx, output/그림/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine

out = Path("output"); fig_dir = out / "그림"; fig_dir.mkdir(exist_ok=True)
S = engine.load_settings()
plt.rcParams.update({"font.family": ["AppleGothic", "Malgun Gothic", "NanumGothic", "sans-serif"], "axes.unicode_minus": False,
                     "figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
C_WEAK, C_NORM, C_STRONG, C_ACC, C_GRAY = "#E07A4F", "#B9BFC9", "#4C7BD9", "#22336B", "#9AA3B0"

D = pd.read_csv(out / "step3_diagnosis_users.csv")
CB = pd.read_csv(out / "step3_diagnosis_categories.csv")
X = pd.read_csv(out / "step4_behavior_types.csv")
names = S["학습유형"]["유형_이름"]

# ---------------- 최종 결과표 (설계 문서 ⑤의 결과표 형식)
T = D.merge(X[["handle", "유형번호"]], on="handle")
T["학습 유형"] = T["유형번호"].map(lambda k: names[str(k - 1)])
T["실효 레이팅"] = np.where(T["실효 레이팅 비고"].fillna("") == "", T["실효 레이팅"].round(0).astype("Int64").astype(str), "계산 불가(" + T["실효 레이팅 비고"].fillna("") + ")")
final = T[["handle", "기준일 레이팅", "실효 레이팅", "전체 판정", "약점 카테고리", "강점 카테고리", "학습 유형"]].rename(columns={"기준일 레이팅": "공식 레이팅"})
final.to_csv(out / "최종_진단결과표.csv", index=False, encoding="utf-8-sig")

# ---------------- 그림 1: 기대 확률 곡선 보정 (2단계)
cal = pd.read_csv(out / "step2_calibration_B.csv")
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(cal))
ax.plot(x, cal["실제_성공비율"] * 100, "o-", color=C_ACC, lw=2, label="실제 성공 비율")
ax.plot(x, cal["예측_평균"] * 100, "s--", color=C_WEAK, lw=2, label="학습한 곡선 예측")
ax.set_xticks(x, cal["구간"]); ax.set_xlabel("문제 난이도 − 레이팅"); ax.set_ylabel("성공 비율 (%)")
ax.set_title("기대 확률 곡선 확인 (B 진단 기간)"); ax.legend(frameon=False); ax.set_ylim(60, 101)
fig.tight_layout(); fig.savefig(fig_dir / "그림1_기대확률_보정.png"); plt.close(fig)

# ---------------- 그림 2: 3단계 진단 결과 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 2]})
oj = D["전체 판정"].value_counts().reindex(["실력 부족", "레이팅 수준", "레이팅 이상"])
axes[0].bar(oj.index, oj.values, color=[C_WEAK, C_NORM, C_STRONG], width=0.6)
for i, v in enumerate(oj.values):
    axes[0].text(i, v, f"{v:,}명\n({v / oj.sum():.0%})", ha="center", va="bottom", fontsize=9)
axes[0].set_title("전체 판정 (공식 레이팅 기준)"); axes[0].set_ylim(0, oj.max() * 1.25)
cc = CB[CB["judge"].isin(["약점", "강점"])].groupby(["category", "judge"]).size().unstack(fill_value=0).reindex(columns=["약점", "강점"])
cc = cc.loc[cc.sum(1).sort_values().index]
y = np.arange(len(cc))
axes[1].barh(y - 0.2, cc["약점"], height=0.4, color=C_WEAK, label="약점")
axes[1].barh(y + 0.2, cc["강점"], height=0.4, color=C_STRONG, label="강점")
axes[1].set_yticks(y, cc.index); axes[1].legend(frameon=False, loc="lower right")
axes[1].set_title("카테고리별 약점·강점 판정 수 (실효 레이팅 기준, 사람 수)")
fig.tight_layout(); fig.savefig(fig_dir / "그림2_진단결과_분포.png"); plt.close(fig)

# ---------------- 그림 3: 실제 사용자 사례
for h, title in [("user12313", "사례 1"), ("user1610", "사례 2")]:
    r = T[T["handle"] == h].iloc[0]
    c = CB[CB["handle"] == h].sort_values("z")
    fig, ax = plt.subplots(figsize=(8, 4.3))
    col = c["judge"].map({"약점": C_WEAK, "강점": C_STRONG, "보통": C_NORM, "판정 보류": "#E6E8EC"})
    ax.barh(c["category"], c["z"], color=col)
    for i, (z, n, sr) in enumerate(zip(c["z"], c["n"], c["solve_rate"])):
        ax.text(z + (0.08 if z >= 0 else -0.08), i, f"z {z:+.2f} · {n}문제 · 정답률 {sr:.0%}", va="center", ha="left" if z >= 0 else "right", fontsize=8)
    for t_ in (-engine.Z_TH, engine.Z_TH):
        ax.axvline(t_, color=C_GRAY, ls="--", lw=1)
    ax.axvline(0, color="#52514e", lw=0.8)
    lim = max(abs(c["z"]).max() + 2.5, 4); ax.set_xlim(-lim, lim)
    ax.set_xlabel("z (실효 레이팅 기준) — 점선 밖이면 약점·강점")
    ax.set_title(f"{title} {h}: 공식 레이팅 {r['기준일 레이팅']:.0f} / 실효 레이팅 {r['실효 레이팅']} / {r['전체 판정']} / {r['학습 유형']}")
    fig.tight_layout(); fig.savefig(fig_dir / f"그림3_{title.replace(' ', '')}_{h}.png"); plt.close(fig)

# ---------------- 그림 4: 학습 유형 프로필
cen = pd.DataFrame(S["학습유형"]["군집_중심"], columns=engine.FEATURES)
fig, ax = plt.subplots(figsize=(11, 3.6))
im = ax.imshow(cen.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(engine.FEATURES)), engine.FEATURES, rotation=30, ha="right")
cnt = X["유형번호"].value_counts()
ax.set_yticks(range(len(cen)), [f"{names[str(k)]} ({cnt[k + 1]:,}명)" for k in range(len(cen))])
for i in range(len(cen)):
    for j in range(len(engine.FEATURES)):
        v = cen.iat[i, j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8, fontweight="bold" if abs(v) >= engine.NAME_TH else "normal")
fig.colorbar(im, ax=ax, fraction=0.025, label="레이팅 구간 안 표준점수")
ax.set_title(f"학습 유형별 행동 지표 (굵은 글씨 = 평균보다 0.5 표준편차 이상 차이) · 안정성 {S['학습유형']['안정성_ARI']} · 레이팅 연관도 {S['학습유형']['레이팅_연관도']}")
fig.tight_layout(); fig.savefig(fig_dir / "그림4_학습유형_프로필.png"); plt.close(fig)

# ---------------- 그림 5: 검증 결과
v1 = pd.read_csv(out / "step6_①판정별.csv", index_col=0)
v2 = pd.read_csv(out / "step6_①같은개수.csv", index_col=0)
v3 = pd.read_csv(out / "step6_②예측.csv", index_col=0)
v4 = pd.read_csv(out / "step6_③레이팅.csv", index_col=0)
keep = pd.read_csv(out / "step6_④유형유지.csv", index_col=0)
same_rate = float((X.set_index("handle")["유형번호"] > 0).mean())  # 자리 맞춤
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
axes[0].bar(v1.index, v1["C_1문제당_실제_기대"] * 100, color=[C_WEAK, C_NORM, C_STRONG], width=0.6)
for i, (v, n) in enumerate(zip(v1["C_1문제당_실제_기대"] * 100, v1["쌍"])):
    axes[0].text(i, v, f"{v:+.2f}%p\n({n:,}쌍)", ha="center", va="bottom", fontsize=8)
axes[0].set_title("① B 판정별 C 1문제당 (실제−기대)"); axes[0].set_ylim(0, v1["C_1문제당_실제_기대"].max() * 100 * 1.5)
axes[1].bar(["우리 진단", "단순 정답률", "무작위"], v2["C에서도 기대 이하(적중률)"] * 100, color=[C_ACC, C_WEAK, C_GRAY], width=0.6)
for i, v in enumerate(v2["C에서도 기대 이하(적중률)"] * 100):
    axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
axes[1].set_title("① 같은 개수로 고른 약점의 C 적중률"); axes[1].set_ylim(0, 60)
axes[2].bar(["(A) 레이팅만", "(B) + 진단"], v3["AUC"], color=[C_GRAY, C_ACC], width=0.55)
axes[2].bar(["실효 레이팅", "공식 레이팅"], v4["AUC"], color=[C_ACC, C_GRAY], width=0.55)
for i, v in enumerate(list(v3["AUC"]) + list(v4["AUC"])):
    axes[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
axes[2].set_ylim(0.65, 0.78); axes[2].set_title("②③ C 성공 예측 AUC"); axes[2].tick_params(axis="x", rotation=20)
per = pd.read_csv(out / "step6_④유형별유지.csv", index_col=0)
xs = np.arange(len(per))
axes[3].bar(xs, per["유지율"] * 100, color=C_ACC, width=0.6, label="같은 유형 유지")
axes[3].scatter(xs, per["무작위 기준(C에서 그 유형 비율)"] * 100, marker="_", s=900, color=C_WEAK, linewidths=3, label="무작위 기준(유형 비율)", zorder=3)
for i, (v, c) in enumerate(zip(per["유지율"] * 100, per["무작위 기준(C에서 그 유형 비율)"] * 100)):
    axes[3].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
axes[3].set_xticks(xs, [n.replace("·", "·\n").replace(" ", "\n", 1) for n in per.index])
axes[3].set_title("④ C에서도 같은 유형 유지 비율"); axes[3].legend(frameon=False, fontsize=8); axes[3].set_ylim(0, 80)
fig.tight_layout(); fig.savefig(fig_dir / "그림5_검증결과.png"); plt.close(fig)

# ---------------- 결과 요약 엑셀
with pd.ExcelWriter(out / "결과요약.xlsx") as w:
    final.to_excel(w, sheet_name="최종 진단 결과표", index=False)
    D["전체 판정"].value_counts().rename_axis("전체 판정").reset_index(name="인원").to_excel(w, sheet_name="전체 판정", index=False)
    CB.groupby(["category", "judge"]).size().unstack(fill_value=0).to_excel(w, sheet_name="카테고리 판정")
    pd.read_csv(out / "step4_type_profile.csv").to_excel(w, sheet_name="학습 유형 평균", index=False)
    v1.to_excel(w, sheet_name="검증① 판정별"); v2.to_excel(w, sheet_name="검증① 같은 개수")
    v3.to_excel(w, sheet_name="검증② 예측"); v4.to_excel(w, sheet_name="검증③ 레이팅"); keep.to_excel(w, sheet_name="검증④ 유형 유지")
print("저장:", out / "최종_진단결과표.csv", out / "결과요약.xlsx", sorted(p.name for p in fig_dir.glob("*.png")))
