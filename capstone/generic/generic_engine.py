"""
범용 진단 엔진 — 어느 채점 사이트의 제출 기록에도 같은 알고리즘을 적용 (재사용 2단계: 새 데이터셋)

순서
  1) 플랫폼 대응표(platforms/*.json)로 열 이름·결과 이름·분야 태그를 공통 형식으로 바꾼다
  2) 팀이 정한 규칙 그대로 사람×문제 표를 만든다 (첫 정답에서 자르기, 기간 끝 관찰 부족 제외 등)
  3) 문제 난이도가 없으면 풀이 기록만으로 '문제 난이도'와 '학생 실력'을 동시에 추정한다
     (문항반응이론 1모수 모형. 기대 확률 곡선은 2단계에서 저장한 것과 같은 식을 그대로 쓴다)
  4) 3단계(실효 레이팅·분야별 약점/강점)와 4단계(학습 유형)를 원래 엔진 함수로 그대로 계산한다
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine  # noqa: E402  (원래 진단 엔진: 기대 확률 곡선·z·판정·행동 지표·유형 배정)

PLATFORM_DIR = Path(__file__).resolve().parent / "platforms"
CANON = {"정답": "OK", "오답": "WRONG_ANSWER", "시간초과": "TIME_LIMIT_EXCEEDED", "메모리초과": "MEMORY_LIMIT_EXCEEDED",
         "런타임에러": "RUNTIME_ERROR", "컴파일에러": "COMPILATION_ERROR", "기타오답": "OTHER_FAIL"}
ERR_COLS = {"WRONG_ANSWER": "n_WA", "TIME_LIMIT_EXCEEDED": "n_TLE", "MEMORY_LIMIT_EXCEEDED": "n_MLE",
            "RUNTIME_ERROR": "n_RE", "COMPILATION_ERROR": "n_CE"}

# ------------------------------------------------------------
# 팀이 정한 기준 (재사용 2단계, 2026-10-29 확정)
#  - 결과·분야 대응표: 기존 분류 그대로 (10개 분야, 정답/오답/시간초과/메모리초과/런타임/컴파일/기타오답)
#  - 실력·난이도: Rasch 모형(1모수 문항반응이론)으로 풀이 기록에서 동시 추정
#  - 공식 레이팅 대비 '전체 판정'은 범용 버전에서 제외 (다른 사이트에는 공식 레이팅이 없음)
#  - 학생 최소 30문제, 분야 최소 5문제는 그대로 + 문제별 최소 응답자(도전 학생) 기준 추가
# ------------------------------------------------------------
MIN_PROBLEM_ATTEMPTS = 5      # [팀 결정] 문제별 최소 응답자: 도전 학생 5명 미만인 문제는 난이도를 믿기 어려워 진단에서 뺌
ABILITY_BANDS = 5             # 학습 유형 표준화용 실력 구간 수 (추정 실력의 5분위. 3·5·6 비교에서 차이 없음)
PRIOR_SD = 600                # 난이도·실력 추정의 안정화(극단값 방지) 정도
CENTER = 1500                 # 추정 실력의 기준점 (숫자 위치만 정함. 판정 결과에는 영향 없음)


def load_platform(name_or_path):
    p = Path(name_or_path)
    if not p.exists():
        p = PLATFORM_DIR / f"{name_or_path}.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------
# 1) 공통 형식으로 바꾸기
# ------------------------------------------------------------
def standardize(df, cfg):
    col = cfg["열"]
    v2c = {v: CANON[k] for k, vs in cfg["결과_대응"].items() for v in vs}
    t2c = {t.strip().lower(): c for c, ts in cfg["분야_대응"].items() for t in ts}
    sep = cfg.get("분야_구분자", ";")

    out = pd.DataFrame({"handle": df[col["학생"]].astype(str), "problem_id": df[col["문제"]].astype(str),
                        "time": pd.to_datetime(df[col["제출시각"]])})
    raw = df[col["결과"]].astype(str).str.strip()
    out["verdict"] = raw.map(lambda v: v2c.get(v, v2c.get(v.split(" (")[0], "EXCLUDE")))   # "런타임 에러 (Segfault)" 같은 꼬리말 허용
    out["verdict_group"] = np.select([out["verdict"] == "OK", out["verdict"] == "COMPILATION_ERROR", out["verdict"] == "EXCLUDE"],
                                     ["success", "behavior_only", "exclude"], "fail")

    def cats(s):
        cs = []
        for t in str(s).split(sep):
            c = t2c.get(t.strip().lower())
            if c and c not in cs:
                cs.append(c)
        return ";".join(cs)
    tags = df[col["분야"]].fillna("") if col.get("분야") else pd.Series("", index=df.index)   # 분야 태그가 없는 사이트(예: AtCoder)도 허용
    uniq = {t: cats(t) for t in tags.unique()}
    out["categories"] = tags.map(uniq)
    out["problem_rating"] = df[col["문제난이도"]].astype(float) if col.get("문제난이도") else np.nan
    out["rating_at_submission"] = df[col["학생레이팅"]].astype(float) if col.get("학생레이팅") else np.nan
    report = {"제출 수": len(out), "제외된 결과": int((out["verdict_group"] == "exclude").sum()),
              "대응표에 없는 결과 이름": sorted(set(raw[out["verdict"] == "EXCLUDE"].unique()))[:20],
              "분야를 못 찾은 제출 비율": round(float((out["categories"] == "").mean()), 4)}
    return out[out["verdict_group"] != "exclude"].reset_index(drop=True), report


# ------------------------------------------------------------
# 2) 사람×문제 표 (0단계와 같은 규칙)
# ------------------------------------------------------------
def make_user_problem(sub, start=None, end=None, short_obs_days=0):
    keys = ["handle", "problem_id"]
    s = sub.sort_values(keys + ["time"], kind="stable")
    first_all = s.groupby(keys, sort=False)["time"].transform("min")
    keep = pd.Series(True, index=s.index)
    if start is not None:
        keep &= first_all >= pd.Timestamp(start)
    if end is not None:
        keep &= first_all < pd.Timestamp(end) - pd.Timedelta(days=short_obs_days)
        keep &= s["time"] < pd.Timestamp(end)
    s = s[keep]
    is_ok = (s["verdict_group"] == "success").astype(int)
    ok_before = is_ok.groupby([s["handle"], s["problem_id"]], sort=False).cumsum() - is_ok
    s = s[ok_before == 0]                                   # 첫 정답에서 자르기

    g = s.groupby(keys, sort=False)
    up = g.agg(problem_rating=("problem_rating", "first"), categories=("categories", "first"),
               first_rating=("rating_at_submission", "first"), n_attempts=("verdict", "size"),
               first_time=("time", "first"), first_verdict=("verdict_group", "first")).reset_index()
    solved_time = s[s["verdict_group"] == "success"].groupby(keys)["time"].first().rename("solved_time")
    up = up.merge(solved_time, on=keys, how="left")
    up["solved"] = up["solved_time"].notna().astype(int)
    up["first_try_ok"] = (up["first_verdict"] == "success").astype(int)
    cnt = s[s["verdict"].isin(ERR_COLS)].groupby(keys + ["verdict"]).size().unstack(fill_value=0).rename(columns=ERR_COLS)
    up = up.merge(cnt, on=keys, how="left")
    for c in ERR_COLS.values():
        up[c] = up[c].fillna(0).astype(int) if c in up.columns else 0
    other = s[(s["verdict_group"] == "fail") & ~s["verdict"].isin(ERR_COLS)].groupby(keys).size().rename("n_OTHER")
    up = up.merge(other, on=keys, how="left")
    up["n_OTHER"] = up["n_OTHER"].fillna(0).astype(int)
    real = s[s["verdict_group"].isin(["success", "fail"])].groupby(keys).size().rename("n_judged")
    up = up.merge(real, on=keys, how="left")
    up["n_judged"] = up["n_judged"].fillna(0).astype(int)
    up = up[up["n_judged"] > 0].drop(columns="first_verdict")       # 컴파일 에러만 있는 문제는 성공·실패 판정에서 제외
    return up.reset_index(drop=True), s


# ------------------------------------------------------------
# 3) 난이도·실력 동시 추정 — Rasch 모형 (1모수 문항반응이론)
#    Rasch: P(맞힘) = 1 / (1 + exp(−(실력 − 난이도)))  (실력·난이도가 같은 눈금 위에 놓임)
#    여기서는 2단계에서 저장한 기대 확률 곡선과 같은 눈금을 쓰기 위해
#      P(맞힘) = 1 / (1 + exp(−(a + b × (난이도 − 실력)/400)))  (a, b는 저장값 그대로)
#    로 계산한다. 이는 Rasch 모형을 레이팅 눈금(1 logit ≈ 400/|b| ≈ 635점)으로 바꿔 쓴 것과 같다.
#    추정 방법: 학생 실력과 문제 난이도를 번갈아 뉴턴법으로 갱신 (결합 최대우도, 극단값 방지용 약한 정규 사전분포)
# ------------------------------------------------------------
def estimate_difficulty(up, settings, iters=60, prior_sd=PRIOR_SD, center=CENTER):
    m = settings["기대확률_모델"]; a, k = m["a"], m["b"] / 400
    u_idx, users = pd.factorize(up["handle"]); j_idx, probs = pd.factorize(up["problem_id"])
    y = up["solved"].to_numpy(float)
    th = np.full(len(users), float(center)); d = np.full(len(probs), float(center))
    v = 1 / prior_sd ** 2
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + k * (d[j_idx] - th[u_idx]))))
        g = np.bincount(u_idx, (y - p) * (-k), len(users)) - (th - center) * v
        h = np.bincount(u_idx, p * (1 - p) * k * k, len(users)) + v
        th += np.clip(g / h, -300, 300)
        p = 1 / (1 + np.exp(-(a + k * (d[j_idx] - th[u_idx]))))
        g = np.bincount(j_idx, (y - p) * k, len(probs)) - (d - center) * v
        h = np.bincount(j_idx, p * (1 - p) * k * k, len(probs)) + v
        d += np.clip(g / h, -300, 300)
    n = np.bincount(j_idx, minlength=len(probs))
    return pd.DataFrame({"problem_id": probs, "추정 난이도": d, "도전 학생 수": n}), pd.Series(th, index=users, name="추정 실력")


# ------------------------------------------------------------
# 4) 진단 (원래 엔진 함수를 그대로 사용)
# ------------------------------------------------------------
def diagnose(up, diff, settings, min_attempts=MIN_PROBLEM_ATTEMPTS):
    """up의 problem_rating이 비어 있으면 추정 난이도로 채운다. 도전 학생 수가 적은 문제는 난이도를 믿기 어려워 진단에서 뺀다."""
    up = up.merge(diff, on="problem_id", how="left")
    if up["problem_rating"].isna().all():
        up["problem_rating"] = up["추정 난이도"]
        up = up[up["도전 학생 수"] >= min_attempts]
    users, cats = [], []
    for h, g in up.groupby("handle", sort=True):
        pr, y = g["problem_rating"].to_numpy(float), g["solved"].to_numpy(float)
        r_eff, note = engine.effective_rating(pr, y, settings)
        row = {"handle": h, "문제 수": len(g), "해결 수": int(y.sum()), "실효 레이팅": round(r_eff, 1), "실효 레이팅 비고": note}
        p_eff = engine.expect_prob(pr, r_eff, settings)
        long = pd.DataFrame({"cat": g["categories"].astype(str).str.split(";"), "y": y, "p": p_eff}).explode("cat")
        long = long[long["cat"] != ""]
        for c, gg in long.groupby("cat"):
            co, ce, cz = engine.z_score(gg["y"].to_numpy(), gg["p"].to_numpy())
            cats.append({"handle": h, "category": c, "n": len(gg), "solved": int(co), "expected": round(ce, 2),
                         "z": round(cz, 2), "judge": engine.judge(len(gg), cz) if not note else "판정 보류"})
        users.append(row)
    D = pd.DataFrame(users).set_index("handle")
    C = pd.DataFrame(cats, columns=["handle", "category", "n", "solved", "expected", "z", "judge"])   # 분야 태그가 없으면 빈 표
    wk = C[C["judge"] == "약점"].groupby("handle")["category"].agg(";".join)
    st = C[C["judge"] == "강점"].groupby("handle")["category"].agg(";".join)
    D["약점 카테고리"], D["강점 카테고리"] = wk.reindex(D.index).fillna(""), st.reindex(D.index).fillna("")
    return D, C, up


def learning_types(up, sub_cut, D, settings, bands=ABILITY_BANDS, saved=None):
    """4단계 행동 지표 → 같은 실력 구간 안에서 표준화 → 저장된 4개 유형 중심에 배정 (중심은 다시 학습하지 않음)
    saved: 이 사이트에서 저장해 둔 구간 경계·표준화 값 (주면 새 학생 몇 명만으로도 같은 기준으로 배정)"""
    up = up.copy()
    eff = D["실효 레이팅"]
    up["first_rating"] = up["handle"].map(eff)                     # '도전 난이도' = 추정 난이도 − 추정 실력
    s = sub_cut[sub_cut["handle"].isin(D.index)]
    gaps = {h: engine.regap_minutes(g) for h, g in s.groupby("handle")}
    X = pd.DataFrame({h: engine.behavior_metrics(g, gaps.get(h, pd.Series(dtype=float))) for h, g in up.groupby("handle")}).T
    if saved:
        edges = saved["_실력_경계"]
    else:
        edges = list(eff.quantile([i / bands for i in range(1, bands)]).round(1))
    labels = [f"실력 {i + 1}구간" for i in range(len(edges) + 1)]
    band = pd.cut(eff.reindex(X.index), [-float("inf")] + edges + [float("inf")], labels=labels).astype(str)
    if saved:
        med = band.map(lambda b: float(pd.Series(saved[b]["재제출_중앙값"]).iloc[0]))
    else:
        med = X.groupby(band)["재제출 간격(분)"].transform("median")
    X["재제출 간격(분)"] = X["재제출 간격(분)"].astype(float).fillna(med)
    T = engine.transform(X[engine.FEATURES].astype(float))
    Z = pd.DataFrame(index=T.index, columns=engine.FEATURES, dtype=float)
    stats = {"_실력_경계": edges}
    for b in sorted(band.unique()):
        m = (band == b).to_numpy()
        if saved:
            mu, sd = pd.Series(saved[b]["평균"]), pd.Series(saved[b]["표준편차"]).replace(0, 1)
            Z.loc[m] = ((T[m] - mu[engine.FEATURES]) / sd[engine.FEATURES]).to_numpy()
            continue
        mu, sd = T[m].mean(), T[m].std(ddof=0).replace(0, 1)
        Z.loc[m] = ((T[m] - mu) / sd).to_numpy()
        stats[b] = {"평균": mu.round(4).to_dict(), "표준편차": sd.round(4).to_dict(), "인원": int(m.sum()),
                    "재제출_중앙값": [round(float(X.loc[m, "재제출 간격(분)"].median()), 3)]}
    lab, names = engine.assign_type(Z.fillna(0.0), settings)
    X["실력 구간"], X["유형번호"], X["학습 유형"] = band, lab + 1, names
    return X, stats
