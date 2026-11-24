"""
0단계. 데이터 정리
  ① 태그 붙이기  ② 태그 → 카테고리  ③ 행 정리  ④ 사람 × 문제 한 줄로 묶기

실행: 터미널에서  python step0_prepare.py
결과: output 폴더에 정리된 파일들이 생깁니다.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# ============================================================
# [설정] 여기만 바꾸면 됩니다
# ============================================================
DATA_FILE = "../usersCodeforcesSubmissionsEnd2024 (2).csv"        # 데이터 파일 이름 (.csv 또는 .xlsx)
CATEGORY_FILE = "category_map.json" # 태그 → 카테고리 묶음표
OUTPUT_DIR = "output"               # 결과가 저장될 폴더

# verdict 분류 (팀이 정한 대로 고치면 됩니다)
SUCCESS = ["OK"]
FAIL = ["WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED",
        "RUNTIME_ERROR", "IDLENESS_LIMIT_EXCEEDED", "PRESENTATION_ERROR",
        "CHALLENGED",               # CHALLENGED = 해킹당함(풀이가 틀렸다는 뜻)
        "PARTIAL"]                  # [팀 결정 3] 부분 점수 = 완전한 정답이 아니므로 실패(기타 오답)
BEHAVIOR_ONLY = ["COMPILATION_ERROR"]  # 실패로 세지 않고 행동 지표에만 사용
# 위 세 목록에 없는 verdict(SKIPPED, TESTING, REJECTED, FAILED, CRASHED 등)는 모두 제외됩니다.

# 기간 (팀 결정)
BASE_DATE = "2024-07-01"            # 기준일 = 진단 기간이 끝나는 날 (※ 팀 확정 필요: 비교 실험과 같은 날짜로 넣어 둠)
DIAG_MONTHS = 6                     # [팀 결정 6] 진단 기간 = 기준일 전 6개월
SHORT_OBS_DAYS = 7                  # [팀 결정 5] 기간 끝 7일 안에 시작한 시도는 맞혔든 못 맞혔든 전부 제외
MIN_USER_PROBLEMS = 30              # 진단 대상 최소 문제 수 (설계 문서·유의사항)

EXCLUDE_TAGS = ["*special"]
API_URL = "https://codeforces.com/api/problemset.problems"
# ============================================================

out = Path(OUTPUT_DIR)
out.mkdir(exist_ok=True)
report = []


def log(msg):
    print(msg)
    report.append(msg)


# ------------------------------------------------------------
# 데이터 읽기
# ------------------------------------------------------------
if not Path(DATA_FILE).exists():
    sys.exit(f"[오류] '{DATA_FILE}' 파일이 없습니다. 이 스크립트와 같은 폴더에 두고, "
             f"맨 위 DATA_FILE에 파일 이름을 정확히 적어 주세요.")

if DATA_FILE.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(DATA_FILE)
else:
    df = pd.read_csv(DATA_FILE)

df.columns = [c.strip() for c in df.columns]
need = ["handle", "rating_at_submission", "problem_rating",
        "id_of_submission_task", "verdict", "time"]
missing = [c for c in need if c not in df.columns]
if missing:
    sys.exit(f"[오류] 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

print("\n데이터 앞부분 5줄:")
print(df[need].head().to_string())
print()
log(f"[시작] 전체 제출 {len(df):,}건, 사용자 {df['handle'].nunique():,}명")

# ------------------------------------------------------------
# ③-1 완전히 같은 행 지우기
# ------------------------------------------------------------
n = len(df)
df = df.drop_duplicates()
log(f"[③] 중복 행 삭제: {n - len(df):,}건")

# ------------------------------------------------------------
# ③-2 time → 날짜
#   숫자면 Unix 시간(초, 너무 크면 밀리초), 글자면 날짜 문자열로 읽습니다.
# ------------------------------------------------------------
t = pd.to_numeric(df["time"], errors="coerce")
if t.notna().mean() > 0.9:
    unit = "ms" if t.median() > 1e11 else "s"
    df["time"] = pd.to_datetime(t, unit=unit, errors="coerce")
else:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
n = len(df)
df = df[df["time"].notna()]
log(f"[③] 날짜로 못 바꾼 행 삭제: {n - len(df):,}건  "
    f"(기간: {df['time'].min()} ~ {df['time'].max()})")

# ------------------------------------------------------------
# ③-3 verdict 분류
# ------------------------------------------------------------
df["verdict"] = df["verdict"].astype(str).str.strip().str.upper()


def verdict_group(v):
    if v in SUCCESS:
        return "success"
    if v in FAIL:
        return "fail"
    if v in BEHAVIOR_ONLY:
        return "behavior_only"
    return "exclude"


df["verdict_group"] = df["verdict"].map(verdict_group)
log("[③] verdict 개수:\n" + df.groupby(["verdict_group", "verdict"]).size().to_string())
n = len(df)
df = df[df["verdict_group"] != "exclude"]
log(f"[③] 채점 미완료 등 제외: {n - len(df):,}건")

# ------------------------------------------------------------
# ③-4 problem_rating 비어 있는 문제 제외
# ------------------------------------------------------------
df["problem_rating"] = pd.to_numeric(df["problem_rating"], errors="coerce")
n = len(df)
df = df[df["problem_rating"].notna() & (df["problem_rating"] > 0)]
log(f"[③] problem_rating 없는 제출 삭제: {n - len(df):,}건")

# ------------------------------------------------------------
# ③-5 rating_at_submission 비었거나 0인 제출 제외
#      유효한 레이팅이 하나도 없는 사용자는 '진단 불가' 목록에 저장
# ------------------------------------------------------------
df["rating_at_submission"] = pd.to_numeric(df["rating_at_submission"], errors="coerce")
valid = df["rating_at_submission"].notna() & (df["rating_at_submission"] > 0)
all_users = set(df["handle"])
n = len(df)
df = df[valid]
no_rating = sorted(all_users - set(df["handle"]))
pd.DataFrame({"handle": no_rating, "status": "진단 불가(레이팅 없음)"}).to_csv(
    out / "users_no_rating.csv", index=False, encoding="utf-8-sig")
log(f"[③] 레이팅 없는 제출 삭제: {n - len(df):,}건, 진단 불가 사용자: {len(no_rating):,}명")

# ------------------------------------------------------------
# ① 문제 번호 나누기: '1520A', '1520_A', '1520/A', '1520-A1' → 대회 1520, 문제 A
# ------------------------------------------------------------
pat = re.compile(r"^\s*(\d+)\s*[/_\-\s]?\s*([A-Za-z]\d*)\s*$")


def split_id(x):
    m = pat.match(str(x))
    return (int(m.group(1)), m.group(2).upper()) if m else (None, None)


parts = df["id_of_submission_task"].map(split_id)
df["contest_id"] = parts.map(lambda p: p[0])
df["problem_index"] = parts.map(lambda p: p[1])
bad = df[df["contest_id"].isna()]
if len(bad):
    log(f"[①] 문제 번호를 못 읽은 행: {len(bad):,}건, 예시: "
        f"{bad['id_of_submission_task'].astype(str).unique()[:5].tolist()}")
df = df[df["contest_id"].notna()].copy()
df["contest_id"] = df["contest_id"].astype(int)
df["problem_id"] = df["contest_id"].astype(str) + df["problem_index"]

# ------------------------------------------------------------
# ① Codeforces 문제 목록 받아서 태그 붙이기 (한 번 받으면 저장해 두고 재사용)
# ------------------------------------------------------------
cache = out / "problems_cache.json"
if cache.exists():
    problems = json.loads(cache.read_text(encoding="utf-8"))
    log("[①] 저장해 둔 문제 목록 사용 (새로 받으려면 output/problems_cache.json 삭제)")
else:
    print("[①] Codeforces에서 문제 목록 받는 중...")
    with urllib.request.urlopen(API_URL, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data.get("status") != "OK":
        sys.exit(f"[오류] Codeforces API 응답 실패: {data.get('comment')}")
    problems = data["result"]["problems"]
    cache.write_text(json.dumps(problems, ensure_ascii=False), encoding="utf-8")
    log(f"[①] 문제 목록 {len(problems):,}개 받음")

tags_of = {f"{p['contestId']}{p['index'].upper()}":
           [t for t in p.get("tags", []) if t not in EXCLUDE_TAGS]
           for p in problems if "contestId" in p}

# ------------------------------------------------------------
# ② 태그 → 카테고리
# ------------------------------------------------------------
cat_map = json.loads(Path(CATEGORY_FILE).read_text(encoding="utf-8"))
tag_to_cat = {tag: cat for cat, tags in cat_map.items() for tag in tags}

all_tags = {t for ts in tags_of.values() for t in ts}
unknown = sorted(all_tags - set(tag_to_cat))
if unknown:
    log(f"[②] 묶음표에 없는 태그 (category_map.json에 추가 필요): {unknown}")


def cats_of(pid):
    # 같은 카테고리는 한 번만 (set)
    return sorted({tag_to_cat[t] for t in tags_of.get(pid, []) if t in tag_to_cat})


prob_cats = {pid: cats_of(pid) for pid in df["problem_id"].unique()}
n = len(df)
df["categories"] = df["problem_id"].map(lambda p: ";".join(prob_cats[p]))
df = df[df["categories"] != ""]
log(f"[①②] 태그(카테고리) 없는 문제의 제출 삭제: {n - len(df):,}건 "
    f"(gym 문제 등 문제 목록에 없는 문제 포함)")

# 문제 × 카테고리 표 (한 문제가 여러 카테고리면 여러 줄)
pc = pd.DataFrame([(p, c) for p, cs in prob_cats.items() for c in cs],
                  columns=["problem_id", "category"])
pc.to_csv(out / "problem_category.csv", index=False, encoding="utf-8-sig")
log("[②] 카테고리별 문제 수:\n" + pc["category"].value_counts().to_string())

# ------------------------------------------------------------
# 정리된 제출 저장
# ------------------------------------------------------------
df = df.sort_values(["handle", "problem_id", "time"])
cols = ["handle", "rating_at_submission", "problem_rating", "problem_id",
        "contest_id", "problem_index", "verdict", "verdict_group", "time", "categories"]
df[cols].to_csv(out / "submissions_clean.csv", index=False, encoding="utf-8-sig")
log(f"[완료] 정리된 제출 {len(df):,}건, 사용자 {df['handle'].nunique():,}명, "
    f"문제 {df['problem_id'].nunique():,}개")


# ------------------------------------------------------------
# ④ 사람 × 문제 한 줄로 묶기
#    [팀 결정 4] 문제는 '처음 제출한 날'이 속한 기간에 넣는다.
#      → 전체 기록에서 첫 제출 시각을 먼저 구하고, 그 시각이 [start, end) 안인 문제만 남긴다.
#        (기간 전에 처음 도전한 문제를 기간 안에 다시 내도 새 문제로 세지 않음)
#      → 계산에는 그 기간(end 전) 안의 제출만 쓴다.
#    [팀 결정 1] 첫 정답(OK)에서 자른다. 첫 정답 이후 제출은 세지 않는다.
#    [팀 결정 2] 레이팅은 그 문제를 처음 제출할 때의 레이팅.
#    [팀 결정 5] 기간 끝(end) 직전 SHORT_OBS_DAYS일 안에 시작한 문제는 전부 뺀다.
# ------------------------------------------------------------
ERR_COLS = {"WRONG_ANSWER": "n_WA", "TIME_LIMIT_EXCEEDED": "n_TLE",
            "MEMORY_LIMIT_EXCEEDED": "n_MLE", "RUNTIME_ERROR": "n_RE",
            "COMPILATION_ERROR": "n_CE"}


def make_user_problem(sub, start=None, end=None, short_obs_days=0):
    keys = ["handle", "problem_id"]
    s = sub.sort_values(keys + ["time"], kind="stable")

    # 문제마다 '처음 제출한 시각' (전체 기록 기준)
    first_all = s.groupby(keys, sort=False)["time"].transform("min")
    keep = pd.Series(True, index=s.index)
    if start is not None:
        keep &= first_all >= pd.Timestamp(start)
    if end is not None:
        keep &= first_all < pd.Timestamp(end) - pd.Timedelta(days=short_obs_days)   # 끝 직전 N일 안에 시작한 문제 제외
        keep &= s["time"] < pd.Timestamp(end)                                       # 기간 안의 제출만 사용
    s = s[keep]

    # 첫 정답에서 자르기: 첫 OK 이후 제출은 버림 (첫 OK 자체는 포함)
    is_ok = (s["verdict_group"] == "success").astype(int)
    ok_before = is_ok.groupby([s["handle"], s["problem_id"]], sort=False).cumsum() - is_ok
    s = s[ok_before == 0]

    g = s.groupby(keys, sort=False)
    up = g.agg(
        problem_rating=("problem_rating", "first"),
        categories=("categories", "first"),
        first_rating=("rating_at_submission", "first"),      # 첫 제출 때 레이팅
        n_attempts=("verdict", "size"),                       # 첫 정답까지의 제출 수 (컴파일 에러 포함)
        first_time=("time", "first"),
        last_time=("time", "last"),
        first_verdict=("verdict_group", "first"),
    ).reset_index()
    solved_time = s[s["verdict_group"] == "success"].groupby(keys)["time"].first().rename("solved_time")
    up = up.merge(solved_time, on=keys, how="left")
    up["solved"] = up["solved_time"].notna().astype(int)                 # 한 번이라도 OK면 1
    up["first_try_ok"] = (up["first_verdict"] == "success").astype(int)  # 첫 제출이 OK면 1
    up = up.drop(columns="first_verdict")

    # 오류 종류별 횟수 (첫 정답 전 제출만)
    cnt = (s[s["verdict"].isin(ERR_COLS)].groupby(keys + ["verdict"]).size()
           .unstack(fill_value=0).rename(columns=ERR_COLS))
    up = up.merge(cnt, on=keys, how="left")
    for c in ERR_COLS.values():
        if c not in up.columns:
            up[c] = 0
        up[c] = up[c].fillna(0).astype(int)
    other_fail = s[(s["verdict_group"] == "fail") & ~s["verdict"].isin(ERR_COLS)].groupby(keys).size().rename("n_OTHER")
    up = up.merge(other_fail, on=keys, how="left")
    up["n_OTHER"] = up["n_OTHER"].fillna(0).astype(int)                  # 기타 오답 (PARTIAL, CHALLENGED 등)
    up["retries"] = up[["n_WA", "n_TLE", "n_MLE", "n_RE", "n_OTHER"]].sum(axis=1)   # 재시도 = 첫 정답 전 오답 수 (CE 제외)
    up["gap"] = up["problem_rating"] - up["first_rating"]               # 난이도 − 첫 제출 레이팅

    # 성공·실패 판정에 쓰는 제출이 하나도 없는(컴파일 에러만 있는) 문제는 표시
    real = s[s["verdict_group"].isin(["success", "fail"])].groupby(keys).size().rename("n_judged")
    up = up.merge(real, on=keys, how="left")
    up["n_judged"] = up["n_judged"].fillna(0).astype(int)
    return up


up = make_user_problem(df)
up.to_csv(out / "user_problem_all.csv", index=False, encoding="utf-8-sig")
log(f"[④] 사람×문제 {len(up):,}줄 (전체 기간 미리보기), "
    f"성공 비율 {up['solved'].mean():.1%}")

# 진단 기간 (기준일 전 DIAG_MONTHS개월) 사람×문제 표
end = pd.Timestamp(BASE_DATE)
start = end - pd.DateOffset(months=DIAG_MONTHS)
upd = make_user_problem(df, start, end, SHORT_OBS_DAYS)
upd = upd[upd["n_judged"] > 0]
cnt_user = upd.groupby("handle").size()
diag_users = cnt_user[cnt_user >= MIN_USER_PROBLEMS].index
upd.to_csv(out / "user_problem_diag.csv", index=False, encoding="utf-8-sig")
log(f"[④] 진단 기간 {start.date()} ~ {end.date()} (끝 {SHORT_OBS_DAYS}일 안에 시작한 문제 제외): "
    f"사람×문제 {len(upd):,}줄, 성공 비율 {upd['solved'].mean():.1%}")
log(f"[④] 진단 대상 ({MIN_USER_PROBLEMS}문제 이상): {len(diag_users):,}명 / 진단 기간에 문제를 푼 사용자 {len(cnt_user):,}명")

(out / "step0_report.txt").write_text("\n".join(report), encoding="utf-8")
print(f"\n끝! '{OUTPUT_DIR}' 폴더를 확인하세요. 요약은 step0_report.txt에 있습니다.")
