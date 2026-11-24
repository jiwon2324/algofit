"""
진단 엔진 — 다른 데이터에도 그대로 적용하는 재사용 함수 모음
(설계 문서 ③ 강점·약점 진단, ④ 학습 유형 분류, ⑤ 알고리즘화 와 1:1로 대응)

모든 기준값은 model/model_settings.json 에서 읽는다. 새 사용자를 진단할 때 다시 학습하지 않는다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS_FILE = Path(__file__).resolve().parent / "model" / "model_settings.json"

# ------------------------------------------------------------
# 팀이 정한 3·4단계 기준 (2026-10-29 확정)
# ------------------------------------------------------------
Z_TH = 2.0                    # 판정선: z ≤ −2 약점, z ≥ 2 강점 (설계 문서 3-2, 유의사항)
MIN_CAT_PROBLEMS = 5          # 카테고리 문제 수 5개 미만 → 판정 보류 (유의사항)
REGAP_MAX_MIN = 6 * 60        # 재제출 간격은 6시간(360분) 이내만 사용 (설계 문서 4-1)
RATING_BANDS = [0, 1200, 1400, 1600, 1900, 2100, 10000]          # Codeforces 공식 등급 경계 (구현 가이드 4단계)
RATING_BAND_NAMES = ["~1199", "1200-1399", "1400-1599", "1600-1899", "1900-2099", "2100+"]   # 2100 이상은 인원이 적어 합침
N_TYPES = 4                   # 학습 유형 수 (유의사항)
NAME_TH = 0.5                 # 군집 중심이 평균보다 0.5 표준편차 이상 높거나 낮은 지표만 보고 이름 (설계 문서 4-2)

# 4단계 행동 지표 6종 (설계 문서 4-1). 오류 비율은 4가지로 나눠 계산
FEATURES = ["평균 제출 수", "첫 제출 정답률", "재제출 간격(분)", "WA 비율", "TLE·MLE 비율", "RE 비율", "CE 비율",
            "도전 난이도", "포기율"]
LOG_FEATURES = ["평균 제출 수", "재제출 간격(분)"]                  # 로그 변환 (설계 문서 4-2)


def load_settings(path=SETTINGS_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_settings(settings, path=SETTINGS_FILE):
    Path(path).write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# 2단계에서 저장한 기대 확률 곡선
# ------------------------------------------------------------
def expect_prob(problem_rating, user_rating, settings, clip=True):
    """확률 = 1 / (1 + exp(-(a + b × (문제 난이도 − 레이팅)/400)))"""
    m = settings["기대확률_모델"]
    d = (np.asarray(problem_rating, float) - np.asarray(user_rating, float)) / 400
    p = 1 / (1 + np.exp(-(m["a"] + m["b"] * d)))
    return np.clip(p, *m["자르기"]) if clip else p


def z_score(solved, p):
    """z = (실제 성공 수 − 기대 성공률 합) / √(Σ 기대 × (1 − 기대))"""
    o, e, v = float(np.sum(solved)), float(np.sum(p)), float(np.sum(p * (1 - p)))
    return o, e, (o - e) / np.sqrt(v) if v > 0 else 0.0


# ------------------------------------------------------------
# 3단계. 실효 레이팅 · 전체 판정 · 카테고리 판정 (설계 문서 3-3)
# ------------------------------------------------------------
def effective_rating(problem_ratings, solved, settings, lo=0.0, hi=4500.0):
    """기대 성공률 합(자르지 않은 곡선)이 실제 성공 수와 같아지는 레이팅을 이분 탐색으로 찾는다.
    모두 맞혔거나 모두 틀리면 해가 없으므로 탐색 범위 끝값을 돌려주고 표시한다."""
    pr = np.asarray(problem_ratings, float)
    target = float(np.sum(solved))
    f = lambda r: expect_prob(pr, r, settings, clip=False).sum() - target
    if f(hi) <= 0:
        return hi, "상한(거의 다 맞힘)"
    if f(lo) >= 0:
        return lo, "하한(거의 다 틀림)"
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2, ""


def judge(n, z):
    if n < MIN_CAT_PROBLEMS:
        return "판정 보류"
    if z <= -Z_TH:
        return "약점"
    if z >= Z_TH:
        return "강점"
    return "보통"


def overall_judge(z):
    if z <= -Z_TH:
        return "실력 부족"
    if z >= Z_TH:
        return "레이팅 이상"
    return "레이팅 수준"


def diagnose_user(up, settings):
    """한 사람의 진단 기간 사람×문제 표(up) → 진단 결과.
    up 컬럼: problem_rating, first_rating, solved, categories(';'로 구분)"""
    pr, fr, y = up["problem_rating"].to_numpy(float), up["first_rating"].to_numpy(float), up["solved"].to_numpy(float)
    # 전체 판정: 공식 레이팅(그 문제 첫 제출 때 레이팅) 기준
    p_off = expect_prob(pr, fr, settings)
    o, e, z_all = z_score(y, p_off)
    r_eff, eff_note = effective_rating(pr, y, settings)
    # 카테고리 판정: 실효 레이팅 기준
    p_eff = expect_prob(pr, r_eff, settings)
    long = pd.DataFrame({"cat": up["categories"].astype(str).str.split(";"), "y": y, "p": p_eff}).explode("cat")
    cats = []
    for c, g in long.groupby("cat"):
        co, ce, cz = z_score(g["y"].to_numpy(), g["p"].to_numpy())
        jd = judge(len(g), cz) if not eff_note else "판정 보류"   # 실효 레이팅을 계산할 수 없으면(모두 맞힘·모두 틀림) 카테고리 판정 보류
        cats.append({"category": c, "n": len(g), "solved": int(co), "expected": round(ce, 2),
                     "solve_rate": round(co / len(g), 3), "z": round(cz, 2), "judge": jd})
    cats = sorted(cats, key=lambda r: r["z"])
    return {
        "n_problems": int(len(up)), "solved": int(o), "expected_official": round(e, 2),
        "official_rating": float(fr[-1]) if len(fr) else None,
        "effective_rating": round(r_eff, 1), "effective_note": eff_note,
        "z_overall": round(z_all, 2), "overall_judge": overall_judge(z_all),
        "weak": [c["category"] for c in cats if c["judge"] == "약점"],
        "strong": [c["category"] for c in cats if c["judge"] == "강점"][::-1],
        "categories": cats,
    }


# ------------------------------------------------------------
# 4단계. 행동 지표 (설계 문서 4-1)
# ------------------------------------------------------------
def regap_minutes(sub_user_problems):
    """제출 기록(첫 정답에서 자른 것)에서 '틀린 뒤 다음 제출까지' 간격(분). 6시간 이내만."""
    s = sub_user_problems.sort_values(["problem_id", "time"], kind="stable")
    prev_t = s.groupby("problem_id")["time"].shift()
    prev_ok = s.groupby("problem_id")["verdict_group"].shift() == "success"
    gap = (s["time"] - prev_t).dt.total_seconds() / 60
    gap = gap[prev_t.notna() & ~prev_ok]
    return gap[(gap >= 0) & (gap <= REGAP_MAX_MIN)]


def behavior_metrics(up, regaps):
    """한 사람의 사람×문제 표 + 재제출 간격 → 지표 6종(오류 비율은 4개 값)"""
    err = up[["n_WA", "n_TLE", "n_MLE", "n_RE", "n_CE", "n_OTHER"]].sum()
    tot = max(err.sum(), 1)
    return {
        "평균 제출 수": up["n_attempts"].mean(),
        "첫 제출 정답률": up["first_try_ok"].mean(),
        "재제출 간격(분)": float(np.median(regaps)) if len(regaps) else np.nan,
        "WA 비율": err["n_WA"] / tot,
        "TLE·MLE 비율": (err["n_TLE"] + err["n_MLE"]) / tot,
        "RE 비율": err["n_RE"] / tot,
        "CE 비율": err["n_CE"] / tot,
        "도전 난이도": (up["problem_rating"] - up["first_rating"]).mean(),
        "포기율": 1 - up["solved"].mean(),
    }


def rating_band(r):
    return pd.cut(pd.Series(r, dtype=float), RATING_BANDS, labels=RATING_BAND_NAMES, right=False).astype(str).to_numpy()


def transform(X):
    X = X.copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    return X


def standardize(X, bands, settings):
    """저장해 둔 레이팅 구간별 평균·표준편차로 표준화"""
    st = settings["학습유형"]["구간별_표준화"]
    Z = pd.DataFrame(index=X.index, columns=FEATURES, dtype=float)
    for b in np.unique(bands):
        m = np.asarray(bands) == b
        mu = pd.Series(st[b]["평균"]); sd = pd.Series(st[b]["표준편차"]).replace(0, 1)
        Z.loc[m, FEATURES] = ((X.loc[m, FEATURES] - mu[FEATURES]) / sd[FEATURES]).to_numpy()
    return Z.fillna(0.0)


def assign_type(Z, settings):
    """저장해 둔 군집 중심 중 가장 가까운 곳에 배정"""
    C = np.array(settings["학습유형"]["군집_중심"], float)
    d = ((Z[FEATURES].to_numpy()[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    names = settings["학습유형"]["유형_이름"]
    return lab, [names[str(k)] for k in lab]
