"""
3단계. 강점·약점 진단 (B 진단 기간)
  - 전체 판정: 공식 레이팅 기준 전체 z → 실력 부족 / 레이팅 수준 / 레이팅 이상
  - 실효 레이팅: 기대 성공률 합이 실제 성공 수와 같아지는 레이팅
  - 카테고리 판정: 실효 레이팅 기준 카테고리 z → 약점 / 보통 / 강점 / 판정 보류
실행: python step3_diagnosis.py   (step1_periods_model.py 실행 후)
"""
import time
from pathlib import Path

import pandas as pd

import engine

t0 = time.time()
out = Path("output")
S = engine.load_settings()
B_end = pd.Timestamp(S["기간"]["B"][1])

up = pd.read_csv(out / "user_problem_B_diag.csv")
# 기준일 레이팅 = 기준일 직전 마지막 제출의 rating_at_submission (구현 가이드 1단계)
sub = pd.read_csv(out / "submissions_clean.csv", usecols=["handle", "rating_at_submission", "time"], parse_dates=["time"])
base_rating = sub[sub["time"] < B_end].sort_values("time").groupby("handle")["rating_at_submission"].last()

rows, cat_rows = [], []
for h, g in up.groupby("handle", sort=True):
    r = engine.diagnose_user(g, S)
    rows.append({"handle": h, "기준일 레이팅": base_rating.get(h), "실효 레이팅": r["effective_rating"],
                 "실효 레이팅 비고": r["effective_note"], "문제 수": r["n_problems"], "해결 수": r["solved"],
                 "기대 해결 수(공식)": r["expected_official"], "전체 z": r["z_overall"], "전체 판정": r["overall_judge"],
                 "약점 카테고리": ";".join(r["weak"]), "강점 카테고리": ";".join(r["strong"])})
    for c in r["categories"]:
        cat_rows.append({"handle": h, **c})

D = pd.DataFrame(rows)
C = pd.DataFrame(cat_rows)
D.to_csv(out / "step3_diagnosis_users.csv", index=False, encoding="utf-8-sig")
C.to_csv(out / "step3_diagnosis_categories.csv", index=False, encoding="utf-8-sig")

rep = [f"[3단계] 진단 대상 {len(D):,}명 (B 기간 {S['기간']['B'][0]} ~ {S['기간']['B'][1]})",
       "[3단계] 전체 판정 (공식 레이팅 기준):\n" + D["전체 판정"].value_counts().to_string(),
       f"[3단계] 실효 레이팅 − 기준일 레이팅: 중앙값 {(D['실효 레이팅'] - D['기준일 레이팅']).median():+.0f}, "
       f"범위 끝값(해 없음) {(D['실효 레이팅 비고'] != '').sum():,}명",
       "[3단계] 카테고리 판정 (실효 레이팅 기준, 사람×카테고리):\n" + C["judge"].value_counts().to_string(),
       f"[3단계] 약점이 1개 이상인 사람 {(D['약점 카테고리'] != '').sum():,}명, 강점이 1개 이상인 사람 {(D['강점 카테고리'] != '').sum():,}명",
       "[3단계] 카테고리별 약점 판정 수:\n" + C[C["judge"] == "약점"]["category"].value_counts().to_string(),
       f"({time.time() - t0:.0f}초)"]
(out / "step3_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
