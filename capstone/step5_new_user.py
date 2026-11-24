"""
5단계. 저장한 기준값으로 새 사용자 진단 (다시 학습하지 않음)
  같은 규칙으로 정리(0단계) → 진단 기간 표(1단계 규칙) → 3단계 진단 → 저장한 값으로 표준화 → 가장 가까운 유형

사용법
  python step5_new_user.py user5060                 # 데이터셋 사용자 (2024-07-01 기준, 분석 결과와 같아야 함)
  python step5_new_user.py --cf tourist             # 실제 Codeforces 아이디 (오늘 기준 최근 6개월, 인터넷 필요)
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import engine

HERE = Path(__file__).resolve().parent
S = engine.load_settings()
src = (HERE / "step0_prepare.py").read_text(encoding="utf-8")
G = {"pd": pd}
exec(src[src.index("ERR_COLS = {"):src.index("up = make_user_problem(df)")], G)        # 0단계의 사람×문제 규칙 그대로
make_user_problem = G["make_user_problem"]
CFG = {}
exec(src[src.index("SUCCESS = "):src.index("EXCLUDE_TAGS")], CFG)                          # 0단계 verdict 분류 그대로
CAT_MAP = json.loads((HERE / "category_map.json").read_text(encoding="utf-8"))
TAG2CAT = {t: c for c, ts in CAT_MAP.items() for t in ts}
PROBLEMS = json.loads((HERE / "output" / "problems_cache.json").read_text(encoding="utf-8"))


def problem_info(problems):
    info = {}
    for p in problems:
        if "contestId" not in p:
            continue
        cats = sorted({TAG2CAT[t] for t in p.get("tags", []) if t in TAG2CAT})
        info[f"{p['contestId']}{str(p['index']).upper()}"] = (p.get("rating"), ";".join(cats))
    return info


def clean_submissions(sub, info):
    """0단계 ③①② 규칙: verdict 분류, 난이도·레이팅·카테고리 없는 제출 제외"""
    vg = sub["verdict"].map(lambda v: "success" if v in CFG["SUCCESS"] else "fail" if v in CFG["FAIL"]
                            else "behavior_only" if v in CFG["BEHAVIOR_ONLY"] else "exclude")
    sub = sub.assign(verdict_group=vg)
    sub = sub[sub["verdict_group"] != "exclude"]
    if "problem_rating" not in sub or sub["problem_rating"].isna().all():
        sub = sub.assign(problem_rating=sub["problem_id"].map(lambda p: info.get(p, (None, ""))[0]))
    sub = sub.assign(categories=sub["problem_id"].map(lambda p: info.get(p, (None, ""))[1]))
    sub = sub[sub["problem_rating"].notna() & (sub["problem_rating"] > 0) & sub["rating_at_submission"].notna()
              & (sub["rating_at_submission"] > 0) & (sub["categories"] != "")]
    return sub


def diagnose(sub, end_date):
    """한 사람의 제출 기록(handle, problem_id, verdict, time, rating_at_submission[, problem_rating]) → 진단 결과"""
    info = problem_info(PROBLEMS)
    sub = clean_submissions(sub, info)
    end = pd.Timestamp(end_date)
    start = end - pd.DateOffset(months=S["기간"]["진단_개월"])
    up = make_user_problem(sub, start, end, S["기간"]["관찰부족_제외_일수"])
    up = up[up["n_judged"] > 0]
    if len(up) < S["최소_문제수"]["진단_대상"]:
        return {"상태": "진단 불가", "이유": f"진단 기간({start.date()} ~ {end.date()}) 문제 수 {len(up)}개 < {S['최소_문제수']['진단_대상']}개"}
    r = engine.diagnose_user(up, S)
    base = sub[sub["time"] < end].sort_values("time")["rating_at_submission"].iloc[-1]
    # 4단계: 행동 지표 → 저장한 구간별 평균·표준편차로 표준화 → 저장한 군집 중심 중 가장 가까운 곳
    s2 = sub[(sub["time"] >= start) & (sub["time"] < end)].merge(up[["problem_id", "solved_time"]], on="problem_id")
    s2 = s2[s2["solved_time"].isna() | (s2["time"] <= s2["solved_time"])]
    m = engine.behavior_metrics(up, engine.regap_minutes(s2))
    X = pd.DataFrame([m])
    band = engine.rating_band([base])
    if np.isnan(X.loc[0, "재제출 간격(분)"]):
        X.loc[0, "재제출 간격(분)"] = float(np.expm1(S["학습유형"]["구간별_표준화"][band[0]]["평균"]["재제출 간격(분)"]))
    Z = engine.standardize(engine.transform(X[engine.FEATURES].astype(float)), band, S)
    k, name = engine.assign_type(Z, S)
    return {"상태": "진단 완료", "진단 기간": f"{start.date()} ~ {end.date()}", "기준일 레이팅": float(base),
            "실효 레이팅": r["effective_rating"] if not r["effective_note"] else f"계산 불가({r['effective_note']})", "전체 판정": r["overall_judge"], "전체 z": r["z_overall"],
            "약점 카테고리": r["weak"], "강점 카테고리": r["strong"], "학습 유형": name[0],
            "유형 근거": S["학습유형"]["이름_근거"][str(int(k[0]))], "문제 수": r["n_problems"], "카테고리": r["categories"]}


def from_dataset(handle):
    sub = pd.read_csv(HERE / "output" / "submissions_clean.csv", parse_dates=["time"])
    sub = sub[sub["handle"] == handle][["handle", "problem_id", "verdict", "time", "rating_at_submission", "problem_rating"]]
    return diagnose(sub, S["기간"]["기준일"])


def from_codeforces(handle):
    """Codeforces 공식 API: 제출 기록 + 레이팅 기록 → 제출 당시 레이팅을 붙여서 진단 (오늘 기준 최근 6개월)"""
    get = lambda url: json.loads(urllib.request.urlopen(url, timeout=60).read().decode())["result"]
    subs = get(f"https://codeforces.com/api/user.status?handle={handle}")
    hist = get(f"https://codeforces.com/api/user.rating?handle={handle}")
    rt = pd.DataFrame(hist)[["ratingUpdateTimeSeconds", "newRating"]].sort_values("ratingUpdateTimeSeconds") if hist else None
    rows = [{"handle": handle, "problem_id": f"{s['problem'].get('contestId')}{str(s['problem'].get('index', '')).upper()}",
             "verdict": s.get("verdict", "TESTING"), "t": s["creationTimeSeconds"], "problem_rating": s["problem"].get("rating")} for s in subs]
    sub = pd.DataFrame(rows).sort_values("t")
    if rt is None or rt.empty:
        return {"상태": "진단 불가", "이유": "레이팅 기록 없음 (대회 경험 없음)"}
    sub = pd.merge_asof(sub, rt, left_on="t", right_on="ratingUpdateTimeSeconds", direction="backward")
    sub["rating_at_submission"] = sub["newRating"]
    sub["time"] = pd.to_datetime(sub["t"], unit="s")
    return diagnose(sub, pd.Timestamp.now().normalize() + pd.Timedelta(days=1))


if __name__ == "__main__":
    args = sys.argv[1:] or ["user5060"]
    t0 = time.time()
    res = from_codeforces(args[1]) if args[0] == "--cf" else from_dataset(args[0])
    cats = res.pop("카테고리", [])
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if cats:
        print(pd.DataFrame(cats).to_string(index=False))
    print(f"({time.time() - t0:.1f}초)")
