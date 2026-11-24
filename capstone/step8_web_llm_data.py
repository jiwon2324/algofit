"""
웹·LLM 연결용 진단 데이터 만들기 — 사용자 한 명당 JSON 한 줄 (LLM 학습 도우미 앱이 읽음)
실행: python step8_web_llm_data.py   (step3·4 실행 후)
결과: output/웹LLM용_사용자별_진단_v2.jsonl  (앱 폴더 옆 분석결과 폴더에도 복사)
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import engine

out = Path("output")
S = engine.load_settings()
names, descs = S["학습유형"]["유형_이름"], S["학습유형"]["유형_설명"]

D = pd.read_csv(out / "step3_diagnosis_users.csv").set_index("handle")
CB = pd.read_csv(out / "step3_diagnosis_categories.csv")
X = pd.read_csv(out / "step4_behavior_types.csv").set_index("handle")
up = pd.read_csv(out / "user_problem_B_diag.csv")

# 난이도 구간별 (공식 레이팅 기준 기대)
up["p_off"] = engine.expect_prob(up["problem_rating"], up["first_rating"], S)
up["dband"] = pd.cut(up["problem_rating"], [0, 1200, 1600, 2000, 2400, 10000],
                     labels=["800-1199", "1200-1599", "1600-1999", "2000-2399", "2400+"], right=False).astype(str)
DB = up.groupby(["handle", "dband"]).agg(n=("solved", "size"), s=("solved", "sum"), e=("p_off", "sum")).reset_index()

# 행동 지표의 레이팅 구간 안 백분위
pct = X[engine.FEATURES].groupby(X["레이팅 구간"]).rank(pct=True)


def cat_row(r):
    return {"tag": r.category, "category": r.category, "attempts": int(r.n), "correct": int(r.solved),
            "expected": float(r.expected), "solve_rate": float(r.solve_rate),
            "vs_expected": round(r.solved / r.expected, 3) if r.expected else None, "z": float(r.z), "judge": r.judge}


rows = []
cat_by_user = {h: g for h, g in CB.groupby("handle")}
db_by_user = {h: g for h, g in DB.groupby("handle")}
for h, d in D.iterrows():
    cats = [cat_row(r) for r in cat_by_user[h].sort_values("z").itertuples()]
    k = int(X.loc[h, "유형번호"]) - 1
    eff_ok = not isinstance(d["실효 레이팅 비고"], str) or d["실효 레이팅 비고"] == ""
    rows.append({
        "handle": h, "period": f"{S['기간']['B'][0]} ~ {S['기간']['B'][1]}",
        "rating": float(d["기준일 레이팅"]), "rating_band": X.loc[h, "레이팅 구간"],
        "effective_rating": round(float(d["실효 레이팅"]), 0) if eff_ok else None,
        "effective_note": "" if eff_ok else d["실효 레이팅 비고"],
        "overall_judge": d["전체 판정"], "z_overall": float(d["전체 z"]),
        "n_problems": int(d["문제 수"]), "solved": int(d["해결 수"]),
        "overall_vs_expected": round(d["해결 수"] / d["기대 해결 수(공식)"], 3),
        "learning_type": names[str(k)], "learning_type_desc": descs[str(k)],
        "weak_tags": [c for c in cats if c["judge"] == "약점"],
        "strong_tags": [c for c in cats if c["judge"] == "강점"][::-1],
        "categories": cats,
        "difficulty_bands": [{"band": r.dband, "attempts": int(r.n), "solve_rate": round(r.s / r.n, 3), "vs_expected": round(r.s / r.e, 3)}
                             for r in db_by_user[h].sort_values("dband", key=lambda c: c.map({"800-1199": 0, "1200-1599": 1, "1600-1999": 2, "2000-2399": 3, "2400+": 4})).itertuples()],
        "behavior": {f: round(float(X.loc[h, f]), 3) for f in engine.FEATURES},
        "behavior_percentile_in_band": {f: round(float(pct.loc[h, f]), 2) for f in engine.FEATURES},
    })

dst = out / "웹LLM용_사용자별_진단_v2.jsonl"
with open(dst, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
app_data = Path(__file__).resolve().parent.parent / "졸업작품_분석_결과물" / "분석결과"
if app_data.exists():
    shutil.copy(dst, app_data / dst.name)
print(f"{len(rows):,}명 저장: {dst}" + (f" (+ {app_data / dst.name})" if app_data.exists() else ""))
