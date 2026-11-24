# %% [markdown]
# # 코딩 문제 풀이 이력 분석 — 레이팅 보정 강점·약점 진단과 학습 유형 군집화
#
# **졸업작품: 코딩 문제 풀이 이력 분석과 LLM을 활용한 맞춤형 학습 지원 시스템 — 분석 노트북**
#
# | 단계 | 내용 | 결과물 |
# |---|---|---|
# | 0 | 환경 설정 · 설정값 | |
# | 1 | 데이터 불러오기 · 문제정보(태그·난이도) 결합 | 그림 1 |
# | 2 | **시도 테이블** 만들기 (한 사람 × 한 문제 = 1행) | `시도테이블_샘플300행.csv` |
# | 3 | 기대 확률 모델 (Elo 공식 vs 학습 모델) | 그림 2, 표 1 |
# | 4 | **레이팅 보정 태그 진단** (기대 대비, z값, 회귀) | 그림 3·4 |
# | 5 | **검증**: 2024 상반기 진단 → 하반기 확인, 단순 정답률과 비교 | 그림 5, 표 2 |
# | 6 | 비교 실험: 정답값 기준 · 기간 배정 규칙 | 표 3·4 |
# | 7 | 행동 지표 · **학습 유형 군집화** | 그림 6·7, 표 5 |
# | 8 | 웹·LLM 연결용 사용자 진단 JSON 내보내기 | `웹LLM용_사용자별_진단.jsonl` |
# | 9 | 발표·책자용 결과 요약 | `결과요약.md`, `결과표.xlsx` |
# | 10 | 결과 대시보드 (브라우저로 여는 HTML) | `결과_대시보드.html` |
# | 11 | 결과 한 번에 내려받기 | `분석결과.zip` |
#
# **실행 방법 (Colab)**: 런타임 → **모두 실행**. 따로 올릴 파일은 없습니다.
# - 데이터는 원본 출처(HuggingFace denkCF/UsersCodeforcesSubmissionsEnd2024, 135MB zip)에서 자동으로 내려받습니다.
#   팀이 가진 `usersCodeforcesSubmissionsEnd2024 (2).csv`와 내용이 완전히 같은 파일입니다(SHA-256 동일 확인).
# - 같이 받은 `문제정보_Codeforces.json`을 Colab 왼쪽 파일 창의 `/content/cf_project/`에 올려 두면 같은 문제정보로 실행되고,
#   없으면 Codeforces 공식 API에서 자동으로 내려받습니다.
# - 결과는 `/content/cf_project/분석결과/`에 저장되고, 마지막 셀이 `분석결과.zip`을 내려받습니다.
# - 구글 드라이브에 저장하고 싶으면 0절의 `USE_DRIVE = True`로 바꾸세요.
#
# **기간 설계**
# - 기대 확률 모델 학습: 2023-07-01 ~ 2023-12-31 (진단·검증 기간과 겹치지 않게)
# - 진단: 2024-01-01 ~ 2024-06-30
# - 검증: 2024-07-01 ~ 데이터 마지막 날(2024-12-01)

# %% [markdown]
# ## 0. 환경 설정

# %%
import os, sys, json, time, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

IN_COLAB = "google.colab" in sys.modules

USE_DRIVE = False   # True: 구글 드라이브 MyDrive/cf_project 에 저장 (드라이브 접근 허용 필요)

if IN_COLAB:
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE = Path("/content/drive/MyDrive/cf_project")
    else:
        BASE = Path("/content/cf_project")
    BASE.mkdir(parents=True, exist_ok=True)
    # 한글 폰트 (처음 한 번 설치)
    if not Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf").exists():
        os.system("apt-get -qq install -y fonts-nanum > /dev/null")
    fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
else:
    BASE = Path(os.environ.get("CF_BASE", "."))

DATA_PATH = Path(os.environ.get("CF_DATA", BASE / "usersCodeforcesSubmissionsEnd2024.csv"))
OUT = BASE / "분석결과"
FIG = OUT / "그림"
FIG.mkdir(parents=True, exist_ok=True)


def set_korean_font():
    names = {f.name for f in fm.fontManager.ttflist}
    for c in ["NanumGothic", "AppleGothic", "Malgun Gothic", "NanumBarunGothic", "Noto Sans CJK KR"]:
        if c in names:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#e6e5e0"
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["axes.axisbelow"] = True


set_korean_font()

# 그림 색 (약점=주황, 보통=회색, 강점=파랑 / 우리 방법=파랑, 단순 정답률=주황, 무작위=회색)
C_WEAK, C_NORMAL, C_STRONG = "#eb6834", "#a3a29c", "#2a78d6"
C_OURS, C_RAW, C_RAND, C_EXTRA = "#2a78d6", "#eb6834", "#a3a29c", "#1baf7a"
JUDGE_COLOR = {"약점": C_WEAK, "보통": C_NORMAL, "강점": C_STRONG, "미경험": "#d6d5cf"}


def show(df):
    try:
        display(df)
    except NameError:
        print(df.to_string())


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


print("데이터:", DATA_PATH, "| 결과 폴더:", OUT)

# %% [markdown]
# ### 설정값
# 발표·책자에 그대로 적을 수 있도록 모든 기준을 한곳에 모았습니다.
# `COLS`만 바꾸면 다른 플랫폼(백준·solved.ac, LeetCode 등) 데이터에도 같은 파이프라인을 쓸 수 있습니다.

# %%
CONFIG = dict(
    # 원본 컬럼 이름 (다른 데이터셋이면 여기만 바꾸면 됨)
    COLS=dict(user="handle", item="id_of_submission_task", rating="rating_at_submission",
              item_rating="problem_rating", verdict="verdict", time="time"),
    # 기간 (UTC)
    TRAIN_START="2023-07-01",   # 기대 확률 모델 학습
    DIAG_START="2024-01-01",    # 진단 시작
    CUTOFF="2024-07-01",        # 진단/검증 기준일
    # 검증 끝 = 데이터 마지막 제출 시각 (자동)
    SHORT_OBS_DAYS=7,           # 기간 끝 직전 7일 안에 시작한 시도는 '관찰 부족'으로 제외
    TARGET="solved",            # 정답값: 'solved'(같은 기간 안 최종 해결, 기본) / 'first_correct'(첫 제출 정답) — 6-1절에서 비교
    PERIOD_RULE="first",        # 시도의 기간 배정: 'first'(첫 제출) / 'last'(마지막 제출) / 'split'(기준일에서 자름)
    P_CLIP=(0.05, 0.95),        # 진단에서 기대 확률 하한·상한 (z값 폭발 방지)
    SMOOTH_K=10,                # 점수 = 차이 합 / (시도 수 + k)
    Z_TH=1.5,                   # z <= -1.5 약점, z >= 1.5 강점
    MIN_TAG_ATT=5,              # 태그 시도 수 < 5 → 미경험
    MIN_USER_ATT=30,            # 진단 기간 시도 문제 < 30 → 진단 제외
    MIN_VALID_ATT=3,            # 검증 기간 태그 시도 수가 이 이상인 쌍만 검증에 사용
    RIDGE_LAMBDA=2.0,           # 사용자별 태그 회귀의 규제 강도
    RATING_BANDS=[-np.inf, 1200, 1400, 1600, 1900, 2100, np.inf],
    RATING_BAND_NAMES=["~1199", "1200-1399", "1400-1599", "1600-1899", "1900-2099", "2100+"],
    DIFF_BANDS=[0, 1200, 1600, 2000, 2400, 10000],
    DIFF_BAND_NAMES=["800-1199", "1200-1599", "1600-1999", "2000-2399", "2400+"],
    SEED=42,
)

# 채점 결과 분류. 여기에 없는 결과(SKIPPED, TESTING, REJECTED, FAILED, CRASHED, SUBMITTED)는 제외
VERDICT_GROUP = {
    "OK": "OK",
    "WRONG_ANSWER": "WA", "PRESENTATION_ERROR": "WA",
    "TIME_LIMIT_EXCEEDED": "TLE", "MEMORY_LIMIT_EXCEEDED": "TLE",   # TLE·MLE = 효율 부족
    "RUNTIME_ERROR": "RE",
    "COMPILATION_ERROR": "CE",
    "PARTIAL": "OTHER", "CHALLENGED": "OTHER", "IDLENESS_LIMIT_EXCEEDED": "OTHER",
}
VG_CODES = {"OK": 0, "WA": 1, "TLE": 2, "RE": 3, "CE": 4, "OTHER": 5}

# 태그 → 상위 범주, 구분력 낮은 태그 (팀 엑셀 '태그 범주' 시트와 동일)
TAG_CATEGORY = {
    "implementation": "구현·기초", "brute force": "구현·기초", "constructive algorithms": "구현·기초",
    "sortings": "구현·기초", "schedules": "구현·기초", "expression parsing": "구현·기초",
    "greedy": "그리디",
    "math": "수학", "number theory": "수학", "combinatorics": "수학", "probabilities": "수학",
    "chinese remainder theorem": "수학", "matrices": "수학", "fft": "수학",
    "dp": "DP", "bitmasks": "비트마스크",
    "graphs": "그래프", "dfs and similar": "그래프", "shortest paths": "그래프", "dsu": "그래프",
    "graph matchings": "그래프", "flows": "그래프", "2-sat": "그래프",
    "trees": "트리", "data structures": "자료구조",
    "binary search": "탐색·분할", "two pointers": "탐색·분할", "ternary search": "탐색·분할",
    "meet-in-the-middle": "탐색·분할", "divide and conquer": "탐색·분할",
    "strings": "문자열", "hashing": "문자열", "string suffix structures": "문자열",
    "geometry": "기하", "games": "게임·인터랙티브", "interactive": "게임·인터랙티브",
    "communication": "게임·인터랙티브",
}
LOW_DISCRIM_TAGS = {"implementation", "brute force", "constructive algorithms"}
EXCLUDED_TAGS = {"*special"}   # 알고리즘 유형이 아니라 특수 문제 표시라서 진단에서 제외

T_TRAIN = int(pd.Timestamp(CONFIG["TRAIN_START"]).timestamp())
T_DIAG = int(pd.Timestamp(CONFIG["DIAG_START"]).timestamp())
T_CUT = int(pd.Timestamp(CONFIG["CUTOFF"]).timestamp())
RESULTS = {}   # 발표·책자용 핵심 숫자를 모아 두는 곳
TABLES = {}    # 엑셀로 내보낼 표

# %% [markdown]
# ## 1. 데이터 불러오기
# 제출 이력 CSV(약 1,760만 건)와 Codeforces 공식 API의 문제정보(태그·난이도)를 결합합니다.

# %%
# 데이터 파일이 없으면 원본 출처(HuggingFace)에서 내려받기 (팀이 가진 CSV와 같은 파일)
DATA_URL = "https://huggingface.co/datasets/denkCF/UsersCodeforcesSubmissionsEnd2024/resolve/main/usersCodeforcesSubmissionsEnd2024.zip"
if not DATA_PATH.exists():
    import urllib.request, zipfile
    zpath = BASE / "usersCodeforcesSubmissionsEnd2024.zip"
    print("데이터 내려받는 중 (135MB)...")
    urllib.request.urlretrieve(DATA_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extract("usersCodeforcesSubmissionsEnd2024.csv", BASE)
    DATA_PATH = BASE / "usersCodeforcesSubmissionsEnd2024.csv"
    zpath.unlink()

t0 = time.time()
C = CONFIG["COLS"]
raw = pd.read_csv(DATA_PATH, engine="pyarrow")
raw = raw.rename(columns={C["user"]: "handle", C["item"]: "pid", C["rating"]: "rating",
                          C["item_rating"]: "prating", C["verdict"]: "verdict", C["time"]: "t"})
raw["ord"] = np.arange(len(raw), dtype=np.int64)   # 같은 초에 제출된 경우 원본 순서를 보존
T_END = int(raw["t"].max())
print(f"불러오기 {time.time()-t0:.1f}초 | 제출 {len(raw):,}건 | 사용자 {raw['handle'].nunique():,}명 | 문제 ID {raw['pid'].nunique():,}개")
print("기간:", pd.to_datetime(raw["t"].min(), unit="s"), "~", pd.to_datetime(T_END, unit="s"))
RESULTS["n_submissions"] = len(raw)
RESULTS["n_users"] = int(raw["handle"].nunique())
RESULTS["data_end"] = str(pd.to_datetime(T_END, unit="s"))

# %%
# Codeforces 공식 문제정보 (처음 한 번 내려받아 저장)
PROB_CACHE = BASE / "문제정보_Codeforces.json"
if not PROB_CACHE.exists():
    import requests
    r = requests.get("https://codeforces.com/api/problemset.problems", timeout=120)
    r.raise_for_status()
    PROB_CACHE.write_text(r.text)
api = json.loads(PROB_CACHE.read_text())["result"]
probs = pd.DataFrame(api["problems"])
probs["pid"] = probs["contestId"].astype(str) + probs["index"].astype(str)
probs = probs.drop_duplicates("pid").set_index("pid")
probs["tags"] = probs["tags"].apply(lambda ts: sorted(t for t in ts if t not in EXCLUDED_TAGS))
solved_cnt = pd.DataFrame(api["problemStatistics"])
solved_cnt["pid"] = solved_cnt["contestId"].astype(str) + solved_cnt["index"].astype(str)
probs["solvedCount"] = solved_cnt.drop_duplicates("pid").set_index("pid")["solvedCount"]

# 난이도: 원본 problem_rating 우선, 없으면 공식 API 난이도
orig_diff = raw.groupby("pid")["prating"].max()
probs["difficulty"] = orig_diff.reindex(probs.index).fillna(probs["rating"])
print(f"공식 문제 {len(probs):,}개 | 난이도 있음 {probs['difficulty'].notna().sum():,} | 태그 있음 {(probs['tags'].str.len()>0).sum():,}")
match = raw["pid"].isin(probs.index).mean()
print(f"제출-공식문제 병합률 {match:.1%} (병합 안 되는 건 Gym 등 비공식 문제 → 제외)")
RESULTS["merge_rate"] = float(match)

# %%
# 그림 1: 데이터 개요
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
yr = pd.to_datetime(raw["t"], unit="s").dt.year.value_counts().sort_index()
axes[0].bar(yr.index.astype(str), yr.values / 1e6, color=C_OURS, width=0.7)
axes[0].set_title("연도별 제출 수 (백만 건)")
axes[0].tick_params(axis="x", rotation=60)
vc = raw["verdict"].value_counts().head(8)
axes[1].barh(vc.index[::-1], vc.values[::-1] / 1e6, color=C_OURS, height=0.6)
axes[1].set_title("채점 결과 분포 (백만 건, 상위 8개)")
for i, v in enumerate(vc.values[::-1]):
    axes[1].text(v / 1e6, i, f" {v/len(raw):.1%}", va="center", fontsize=9)
savefig(fig, "그림1_데이터_개요")

# %% [markdown]
# ## 2. 시도 테이블 만들기 — "제출 한 건이 아니라, 한 사람이 한 문제를 시도한 것을 한 단위로"
#
# | 항목 | 기준 | 이유 |
# |---|---|---|
# | 해결 여부 | 같은 기간 안에 OK가 한 번이라도 있으면 해결 | 결국 풀어냈는지가 실력에 가장 가깝다 |
# | 첫 제출 정답 | 맨 처음 제출이 OK인지 | 한 번에 맞히는 신중함 |
# | 재시도 횟수 | 첫 OK 전까지 틀린 제출 수 (**CE 제외**) | CE는 언어 선택·오타 등 실력과 거리가 멀고, Codeforces도 벌점을 주지 않음 |
# | 오류 유형 | **첫 OK 전까지의 오답만** 집계 (CE는 따로 비율로) | 첫 OK 이후 제출은 더 빠른 풀이 실험이지 못 푼 것이 아님 |
# | 레이팅·기간 | 첫 제출 시점 기준 | 문제를 시작할 때의 실력과 시점으로 고정 |
#
# **기간 배정 규칙 (팀 논의: 첫 제출 기준 vs 마지막 제출 기준)** — 기본값은 **첫 제출 기준**입니다.
# 1. 기준일 시점에 이미 알 수 있는 정보만으로 정해진다 (마지막 제출 시각은 미래를 봐야 알 수 있음 → 누수).
# 2. 레이팅·시점을 '시작할 때'로 고정하는 다른 항목 기준과 일관된다.
# 3. 마지막 제출 기준으로 하면 오래 걸린(어려운) 시도가 검증 기간으로 넘어가 진단 기간 해결률이 부풀려진다.
#
# 6절에서 세 규칙(첫 제출 / 마지막 제출 / 기준일에서 자르기)을 실제 데이터로 비교해 이 선택을 뒷받침합니다.
# 각 시도의 모든 지표는 **같은 기간 안의 제출만**으로 계산합니다(진단 기간에 시작한 문제의 기준일 이후 제출은 버림).

# %%
t0 = time.time()
sub = raw[raw["verdict"].isin(VERDICT_GROUP.keys()) & raw["pid"].isin(probs.index)].copy()
RESULTS["n_sub_used"] = len(sub)
sub["vg"] = sub["verdict"].map(VERDICT_GROUP).map(VG_CODES).astype(np.int8)
sub["h"] = sub["handle"].astype("category").cat.codes.astype(np.int64)
HANDLES = sub["handle"].astype("category").cat.categories
sub["p"] = pd.Categorical(sub["pid"], categories=probs.index).codes.astype(np.int64)
# 시간순 정렬 (같은 초는 원본 순서: 원본은 최신순이라 ord가 클수록 먼저 제출)
sub = sub.sort_values(["h", "t", "ord"], ascending=[True, True, False], kind="stable")
# 레이팅 결측은 같은 사용자의 직전 레이팅으로 채움 (미래 값은 쓰지 않음)
sub["rating"] = sub.groupby("h", sort=False)["rating"].ffill()
# 제출이 속한 기간: 0=모델학습, 1=진단, 2=검증, -1=그 이전
sub["per"] = np.select([sub["t"] < T_TRAIN, sub["t"] < T_DIAG, sub["t"] < T_CUT], [-1, 0, 1], 2).astype(np.int8)
sub = sub[["h", "p", "t", "vg", "rating", "per"]].reset_index(drop=True)
NP = len(probs)
P_DIFF = probs["difficulty"].to_numpy(dtype=float)
P_NTAGS = probs["tags"].str.len().to_numpy()
PERIOD_END = {0: T_DIAG, 1: T_CUT, 2: T_END}
PERIOD_NAME = {0: "모델학습", 1: "진단", 2: "검증"}
print(f"사용 제출 {len(sub):,}건 정리 {time.time()-t0:.1f}초")
print(sub.groupby("per").size().rename(index={-1: "이전", **PERIOD_NAME}))


# %%
def build_attempts(sub, rule="first"):
    """제출 → 시도 테이블(사용자 × 문제).

    rule='first' : 시도의 기간 = 첫 제출이 속한 기간. 같은 기간 안 제출만 사용.
    rule='last'  : 시도의 기간 = 마지막 제출이 속한 기간. 그 시도의 모든 제출 사용 (비교 실험용).
    rule='split' : 기준일마다 시도를 잘라 기간별로 별도 시도로 취급 (비교 실험용).
    반환: (attempts DataFrame, 재제출 간격 DataFrame)
    """
    key = sub["h"].to_numpy() * NP + sub["p"].to_numpy()
    order = np.argsort(key, kind="stable")              # 사용자·문제별, 그 안은 시간순 유지
    k = key[order]
    t = sub["t"].to_numpy()[order]
    vg = sub["vg"].to_numpy()[order]
    per = sub["per"].to_numpy()[order]
    rating = sub["rating"].to_numpy()[order]

    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    if rule == "first":
        aper = per[start][gid]
        keep = (per == aper) & (aper >= 0)
    elif rule == "last":
        end = np.r_[k[1:] != k[:-1], True]
        aper = per[end][gid]
        keep = aper >= 0
    elif rule == "split":
        aper = per
        keep = per >= 0
    else:
        raise ValueError(rule)
    k, t, vg, aper, rating = k[keep], t[keep], vg[keep], aper[keep], rating[keep]

    start = np.r_[True, (k[1:] != k[:-1]) | (aper[1:] != aper[:-1])]
    sidx = np.flatnonzero(start)
    gid = np.cumsum(start) - 1
    ok = vg == 0
    cs = np.cumsum(ok)
    ok_before = (cs - ok) - (cs - ok)[sidx][gid]       # 같은 시도 안에서 이 제출 이전의 OK 수
    pre = ok_before == 0                                 # 첫 OK까지(포함)의 제출
    pre_wrong = pre & ~ok                                # 첫 OK 전 오답
    first_ok = ok & pre

    def gsum(mask):
        return np.add.reduceat(mask.astype(np.int32), sidx)

    att = pd.DataFrame({
        "h": (k[sidx] // NP).astype(np.int32),
        "p": (k[sidx] % NP).astype(np.int32),
        "period": aper[sidx],
        "start_t": t[sidx],
        "rating": rating[sidx],
        "n_subs": np.diff(np.r_[sidx, len(k)]),
        "n_subs_to_ok": gsum(pre),
        "solved": (gsum(ok) > 0).astype(np.int8),
        "first_correct": ok[sidx].astype(np.int8),
        "wa": gsum(pre_wrong & (vg == 1)),
        "tle": gsum(pre_wrong & (vg == 2)),
        "re": gsum(pre_wrong & (vg == 3)),
        "ce": gsum(pre_wrong & (vg == 4)),
        "other": gsum(pre_wrong & (vg == 5)),
    })
    att["retries"] = att["wa"] + att["tle"] + att["re"] + att["other"]   # CE 제외
    fok_t = np.where(first_ok, t, 0)
    att["first_ok_t"] = np.maximum.reduceat(fok_t, sidx)
    att.loc[att["solved"] == 0, "first_ok_t"] = 0
    att["difficulty"] = P_DIFF[att["p"]]
    att["n_tags"] = P_NTAGS[att["p"]]
    att["gap"] = att["difficulty"] - att["rating"]          # 난이도 − 레이팅 (양수 = 도전)
    pend = att["period"].map(PERIOD_END).to_numpy()
    att["short_obs"] = (pend - att["start_t"] < CONFIG["SHORT_OBS_DAYS"] * 86400).astype(np.int8)
    att["in_model"] = (att["difficulty"].notna() & att["rating"].notna() & (att["short_obs"] == 0)).astype(np.int8)

    # 재제출 간격: 같은 시도 안, 첫 OK까지의 연속 제출 사이 간격(분)
    cont = ~start & pre
    gaps = pd.DataFrame({"h": (k[cont] // NP).astype(np.int32), "period": aper[cont],
                         "gap_min": (t[cont] - t[np.flatnonzero(cont) - 1]) / 60.0})
    return att, gaps


t0 = time.time()
att, gaps = build_attempts(sub, CONFIG["PERIOD_RULE"])
print(f"시도 {len(att):,}개 만들기 {time.time()-t0:.1f}초")
summary = att.groupby("period").agg(
    시도수=("solved", "size"), 사용자수=("h", "nunique"), 해결률=("solved", "mean"),
    첫제출정답률=("first_correct", "mean"), 평균재시도=("retries", "mean"),
    관찰부족비율=("short_obs", "mean"), 모델포함비율=("in_model", "mean")).rename(index=PERIOD_NAME)
show(summary)
TABLES["시도 테이블 요약"] = summary.reset_index()

# 관찰 부족 제외 근거: 기간 끝 직전 7일 안에 시작한 시도는 해결률이 낮게 잘린다
so = att[att["period"] >= 1].groupby(["period", "short_obs"])["solved"].mean().unstack()
so.index = so.index.map(PERIOD_NAME)
so.columns = ["7일 이전 시작", "마지막 7일 안 시작"]
print("\n관찰 부족 효과 (해결률):\n", so.round(3))
TABLES["관찰 부족 효과"] = so.reset_index()

# 샘플 저장 (발표 화면용)
smp = att[att["period"] == 1].head(300).copy()
smp.insert(0, "handle", HANDLES[smp["h"]])
smp.insert(1, "problem", probs.index[smp["p"]])
smp["tags"] = probs["tags"].to_numpy()[smp["p"]]
smp["start_time"] = pd.to_datetime(smp["start_t"], unit="s")
smp.drop(columns=["h", "p"]).to_csv(OUT / "시도테이블_샘플300행.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# ## 3. 기대 확률 모델 — "이 레이팅이면 이 난이도 문제를 풀 확률은?"
#
# - **Elo 공식**: p = 1 / (1 + 10^((난이도 − 레이팅)/400)) — Codeforces 레이팅 체계의 기본 가정
# - **Elo 보정**: 같은 형태지만 척도·절편을 데이터로 학습 (로지스틱 회귀, 변수 1개)
# - **학습 모델**: 로지스틱 회귀 (난이도−레이팅, 그 제곱, 레이팅, 난이도, 태그 수)
#
# 모델은 2023 하반기로 **학습**, 2024 상반기(진단)로 **선택**, 2024 하반기(검증)에서 **최종 확인**합니다.
# 비교 지표: AUC(구분력), log-loss·Brier(확률 정확도, 낮을수록 좋음), 보정표(예측 확률 = 실제 비율인지).

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def model_features(d):
    x = (d["difficulty"].to_numpy() - d["rating"].to_numpy()) / 400.0
    return np.column_stack([x, x ** 2, d["rating"].to_numpy() / 1000, d["difficulty"].to_numpy() / 1000,
                            np.minimum(d["n_tags"].to_numpy(), 8)])


def elo_p(d):
    return 1.0 / (1.0 + 10 ** ((d["difficulty"].to_numpy() - d["rating"].to_numpy()) / 400.0))


def fit_expectation_models(att, target, seed=CONFIG["SEED"]):
    tr = att[(att["period"] == 0) & (att["in_model"] == 1)]
    if len(tr) > 3_000_000:
        tr = tr.sample(3_000_000, random_state=seed)
    y = tr[target].to_numpy()
    m_fit = LogisticRegression(max_iter=200).fit(model_features(tr)[:, :1], y)
    m_full = LogisticRegression(max_iter=500).fit(model_features(tr), y)
    return {
        "Elo 공식(400)": lambda d: elo_p(d),
        "Elo 보정(학습)": lambda d: m_fit.predict_proba(model_features(d)[:, :1])[:, 1],
        "로지스틱 회귀(학습)": lambda d: m_full.predict_proba(model_features(d))[:, 1],
    }, len(tr)


def eval_probs(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    base = np.full_like(p, y.mean())
    return dict(AUC=roc_auc_score(y, p), logloss=log_loss(y, p), Brier=brier_score_loss(y, p),
                Brier개선율=1 - brier_score_loss(y, p) / brier_score_loss(y, base), 시도수=len(y))


def calibration_table(y, p, bins=np.linspace(0, 1, 11)):
    b = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    df = pd.DataFrame({"b": b, "y": y, "p": p}).groupby("b").agg(시도수=("y", "size"), 평균기대=("p", "mean"), 실제해결률=("y", "mean"))
    df.index = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in df.index]
    return df


t0 = time.time()
TARGET = CONFIG["TARGET"]
models, n_train = fit_expectation_models(att, TARGET)
rows = []
for per in (1, 2):
    d = att[(att["period"] == per) & (att["in_model"] == 1)]
    for name, f in models.items():
        rows.append({"기간": PERIOD_NAME[per], "모델": name, **eval_probs(d[TARGET].to_numpy(), f(d))})
model_cmp = pd.DataFrame(rows)
print(f"학습 시도 {n_train:,}개, {time.time()-t0:.1f}초")
print(model_cmp.round(4).to_string(index=False))
TABLES["표1 기대확률 모델 비교"] = model_cmp

# 진단 기간 log-loss 기준으로 모델 선택 (검증 기간은 보지 않고 선택)
best_name = model_cmp[model_cmp["기간"] == "진단"].sort_values("logloss").iloc[0]["모델"]
expect = models[best_name]
print("선택된 기대 확률 모델:", best_name)
RESULTS["expect_model"] = best_name
RESULTS["model_cmp_valid"] = model_cmp[model_cmp["기간"] == "검증"].set_index("모델")[["AUC", "logloss", "Brier"]].round(4).to_dict("index")

# %%
# 그림 2: 보정 곡선 (검증 기간)
d = att[(att["period"] == 2) & (att["in_model"] == 1)]
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--", label="완벽한 보정")
for (name, f), col in zip(models.items(), [C_RAW, C_EXTRA, C_OURS]):
    ct = calibration_table(d[TARGET].to_numpy(), f(d))
    ax.plot(ct["평균기대"], ct["실제해결률"], marker="o", ms=5, lw=2, color=col, label=name)
ax.set_xlabel("기대 확률 (모델 예측)")
ax.set_ylabel("실제 해결률")
ax.set_title("기대 확률 보정 곡선 (검증 기간)")
ax.legend(frameon=False)
savefig(fig, "그림2_기대확률_보정곡선")
TABLES["기대확률 보정표(검증)"] = calibration_table(d[TARGET].to_numpy(), expect(d)).reset_index()

# %% [markdown]
# ## 4. 레이팅 보정 태그 진단
#
# **문제**: 레이팅 1000인 사람은 모든 태그의 정답률이 낮아서, 단순 정답률로 보면 전부 약점이 된다.
# **해결**: 레이팅 차이로 시도마다 기대 확률 p를 계산하고, 태그별로 *기대보다 얼마나 덜(더) 풀었는지*를 본다.
#
# | 지표 | 계산 | 뜻 |
# |---|---|---|
# | 기대 대비 | 실제 해결 수 ÷ 기대 해결 수(Σp) | 100%보다 낮으면 레이팅에 비해 못 푼다 |
# | 차이 | Σ(y − p) | 기대보다 몇 문제 더/덜 풀었나 |
# | z값 | Σ(y−p) ÷ √Σp(1−p) | 우연으로 설명되는 정도를 걸러냄 |
# | 점수(평활화) | Σ(y−p) ÷ (n + k) | 시도가 적은 태그는 0 쪽으로 당김 |
# | 가중 차이 | Σ(y−p)/태그 수 | 여러 태그 문제의 몫을 1/태그 수로 나눔 |
# | 회귀 계수 | logit P = logit(p) + Σβ_태그 | 여러 태그 효과를 회귀로 동시에 분리 |
#
# **판정**: 시도 < 5 → 미경험, z ≤ −1.5 → 약점, z ≥ 1.5 → 강점, 나머지 보통.

# %%
ptags = probs["tags"].reset_index(drop=True).explode().dropna().rename("tag")
PTAGS = pd.DataFrame({"p": ptags.index.astype(np.int32), "tag": ptags.values})
PTAGS["category"] = PTAGS["tag"].map(TAG_CATEGORY).fillna("기타")
ALL_TAGS = sorted(PTAGS["tag"].unique())
TAG_IDX = {t: i for i, t in enumerate(ALL_TAGS)}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def add_expectation(att_part, target, expect_fn):
    """시도마다 기대 확률 p(레이팅·난이도로 계산)를 붙인다."""
    d = att_part[att_part["in_model"] == 1].copy()
    d["y"] = d[target].astype(float)
    d["pe"] = np.clip(expect_fn(d), 1e-4, 1 - 1e-4)   # pe = 기대 확률 (p는 문제 번호 컬럼)
    return d


def judge(n, z):
    th, mn = CONFIG["Z_TH"], CONFIG["MIN_TAG_ATT"]
    return np.where(n < mn, "미경험", np.where(z <= -th, "약점", np.where(z >= th, "강점", "보통")))


def tag_long(d, level="tag"):
    """시도 1건을 태그별로 펼친 표 (기대 확률은 하한·상한으로 제한)"""
    pc = np.clip(d["pe"].to_numpy(), *CONFIG["P_CLIP"])
    y = d["y"].to_numpy()
    base = pd.DataFrame({"aid": d.index.to_numpy(), "h": d["h"].to_numpy(), "p": d["p"].to_numpy(),
                         "y": y, "pc": pc, "resid": y - pc, "var": pc * (1 - pc), "n_tags": d["n_tags"].to_numpy()})
    long = base.merge(PTAGS, on="p")
    if level == "category":
        long = long.drop_duplicates(["aid", "category"])
    return long


def diagnose(d, level="tag"):
    """사용자 × 태그(또는 상위 범주) 진단표"""
    key = "tag" if level == "tag" else "category"
    long = tag_long(d, level)
    long["wresid"] = long["resid"] / long["n_tags"].clip(lower=1)
    g = long.groupby(["h", key], sort=False).agg(n=("y", "size"), solved=("y", "sum"), expected=("pc", "sum"),
                                                 diff=("resid", "sum"), wdiff=("wresid", "sum"), var=("var", "sum")).reset_index()
    g["raw_rate"] = g["solved"] / g["n"]
    g["ratio"] = g["solved"] / g["expected"]
    g["z"] = g["diff"] / np.sqrt(g["var"].clip(lower=1e-9))
    g["score"] = g["diff"] / (g["n"] + CONFIG["SMOOTH_K"])
    g["judge"] = judge(g["n"].to_numpy(), g["z"].to_numpy())
    return g


def ridge_tag_regression(d, lam=CONFIG["RIDGE_LAMBDA"], iters=8):
    """사용자별 로지스틱 회귀: logit P(해결) = logit(p_기대) + Σ β_태그 · [태그 포함]
    여러 태그가 붙은 문제에서 각 태그의 효과를 동시에 분리한다."""
    long = tag_long(d)
    long["ti"] = long["tag"].map(TAG_IDX)
    long = long.sort_values(["h", "aid"])
    T = len(ALL_TAGS)
    out = []
    a_h, a_id, a_ti = long["h"].to_numpy(), long["aid"].to_numpy(), long["ti"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, a_h[1:] != a_h[:-1], True])
    y_all = d["y"]
    pcl = np.clip(d["pe"], *CONFIG["P_CLIP"])
    o_all = np.log(pcl / (1 - pcl))
    for s, e in zip(bounds[:-1], bounds[1:]):
        ids, inv = np.unique(a_id[s:e], return_inverse=True)
        X = np.zeros((len(ids), T))
        X[inv, a_ti[s:e]] = 1.0
        used = X.sum(0) > 0
        X = X[:, used]
        y = y_all.loc[ids].to_numpy()
        off = o_all.loc[ids].to_numpy()
        b = np.zeros(X.shape[1])
        for _ in range(iters):
            mu = sigmoid(off + X @ b)
            H = (X * (mu * (1 - mu))[:, None]).T @ X + lam * np.eye(len(b))
            b = b + np.linalg.solve(H, X.T @ (y - mu) - lam * b)
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        out.append(pd.DataFrame({"h": a_h[s], "tag": np.array(ALL_TAGS)[used], "beta": b, "z_reg": b / se}))
    return pd.concat(out, ignore_index=True)


def run_diagnosis(att_x, target, expect_fn, with_regression=False):
    """진단 기간 시도 → (시도표, 태그 진단, 범주 진단)"""
    d = add_expectation(att_x[att_x["period"] == 1], target, expect_fn)
    un = d.groupby("h").size()
    d = d[d["h"].isin(un[un >= CONFIG["MIN_USER_ATT"]].index)]
    dt = diagnose(d, "tag")
    dc = diagnose(d, "category")
    if with_regression:
        dt = dt.merge(ridge_tag_regression(d), on=["h", "tag"], how="left")
        dt["judge_reg"] = judge(dt["n"].to_numpy(), dt["z_reg"].fillna(0).to_numpy())
    return d, dt, dc


t0 = time.time()
dg, diag_tag, diag_cat = run_diagnosis(att, TARGET, expect, with_regression=True)
DIAG_USERS = pd.Index(dg["h"].unique())
print(f"진단 대상 사용자 {len(DIAG_USERS):,}명 / 진단 기간 시도 {len(dg):,}개 / 사용자×태그 {len(diag_tag):,}쌍 ({time.time()-t0:.1f}초)")
RESULTS["n_diag_users"] = len(DIAG_USERS)
RESULTS["n_diag_attempts"] = len(dg)
jd = pd.DataFrame({"기대 대비 z": diag_tag["judge"].value_counts(normalize=True),
                   "태그 회귀": diag_tag["judge_reg"].value_counts(normalize=True)})
print("판정 분포:\n", jd.round(3))
RESULTS["judge_dist"] = jd.round(4).to_dict()

# %% [markdown]
# ### 왜 레이팅 보정이 필요한가 (발표 3번 슬라이드용)
# 같은 태그라도 단순 정답률과 레이팅 기대 대비는 레이팅 구간마다 다르게 보인다.
# 단순 정답률은 **문제 난이도와 실력 수준**이 섞인 값이고, 기대 대비는 **같은 레이팅 안에서의 상대적 강약**이다.

# %%
last_rating = dg.sort_values("start_t").groupby("h")["rating"].last()
BAND = pd.cut(last_rating, CONFIG["RATING_BANDS"], labels=CONFIG["RATING_BAND_NAMES"], right=False)
show_tags = ["dp", "greedy", "math"]
dt = diag_tag[diag_tag["tag"].isin(show_tags) & (diag_tag["n"] >= CONFIG["MIN_TAG_ATT"])].copy()
dt["band"] = pd.Categorical(BAND.reindex(dt["h"]).to_numpy(), categories=CONFIG["RATING_BAND_NAMES"], ordered=True)
pv_raw = dt.groupby(["band", "tag"], observed=True)[["solved", "n"]].sum().eval("solved / n").unstack()[show_tags]
pv_ratio = dt.groupby(["band", "tag"], observed=True)[["solved", "expected"]].sum().eval("solved / expected").unstack()[show_tags]
print("레이팅 구간별 단순 해결률\n", pv_raw.round(3))
print("레이팅 구간별 기대 대비\n", pv_ratio.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, pv, title in [(axes[0], pv_raw, "단순 해결률"), (axes[1], pv_ratio, "기대 대비 (실제 ÷ 레이팅 기대)")]:
    for i, tg in enumerate(show_tags):
        ax.plot(pv.index.astype(str), pv[tg], marker="o", lw=2, ms=5, color=[C_OURS, C_RAW, C_EXTRA][i], label=tg)
    ax.set_title(title)
    ax.set_xlabel("레이팅 구간")
    ax.legend(frameon=False)
axes[1].axhline(1, color="#8a8984", lw=1, ls="--")
axes[0].set_ylabel("비율")
savefig(fig, "그림3_레이팅구간별_해결률과_기대대비")
TABLES["레이팅 구간별 단순 해결률"] = pv_raw.reset_index()
TABLES["레이팅 구간별 기대 대비"] = pv_ratio.reset_index()

# %%
# 발표용 실제 사용자 사례 (실제 데이터에서 규칙으로 자동 선택)
# 규칙: 최근 레이팅 900~1250 사용자 중 dp(12문제 이상)는 '약점', greedy(15문제 이상)는 '강점'으로 판정된 사람.
#       그중 약점 태그 수가 가장 적은 사람, 같으면 레이팅 1000에 가장 가까운 사람 (대본의 "레이팅 1000인 사람" 예시)
cand = diag_tag[diag_tag["h"].isin(last_rating[(last_rating >= 900) & (last_rating < 1250)].index)
                & (diag_tag["n"] >= CONFIG["MIN_TAG_ATT"])]
dp_weak = set(cand[(cand["tag"] == "dp") & (cand["n"] >= 12) & (cand["judge"] == "약점")]["h"])
gr_strong = set(cand[(cand["tag"] == "greedy") & (cand["n"] >= 15) & (cand["judge"] == "강점")]["h"])
ok_users = sorted(dp_weak & gr_strong)
if not ok_users:   # 조건을 모두 만족하는 사람이 없으면 dp 약점인 사람으로
    ok_users = sorted(dp_weak) or sorted(cand["h"].unique())
n_weak = cand[cand["h"].isin(ok_users)].groupby("h")["judge"].apply(lambda s: (s == "약점").sum())
pick = pd.DataFrame({"weak": n_weak, "dist": (last_rating.reindex(n_weak.index) - 1000).abs()}).sort_values(["weak", "dist"])
EXAMPLE_H = int(pick.index[0])
print(f"조건(dp 약점 + greedy 강점, 레이팅 900~1250)을 만족한 실제 사용자 {len(dp_weak & gr_strong)}명 중 선택")
ex = diag_tag[(diag_tag["h"] == EXAMPLE_H) & (diag_tag["n"] >= CONFIG["MIN_TAG_ATT"])].sort_values("ratio")
ex_show = ex[["tag", "n", "solved", "raw_rate", "expected", "ratio", "z", "judge"]].rename(columns={
    "tag": "태그", "n": "시도", "solved": "해결", "raw_rate": "단순 해결률", "expected": "기대 해결 수",
    "ratio": "기대 대비", "z": "z값", "judge": "판정"})
print(f"실제 사용자 사례 {HANDLES[EXAMPLE_H]} (최근 레이팅 {last_rating[EXAMPLE_H]:.0f})")
print(ex_show.round(2).to_string(index=False))
TABLES["예시 사용자 태그 진단"] = ex_show
RESULTS["example_user"] = dict(handle=str(HANDLES[EXAMPLE_H]), rating=float(last_rating[EXAMPLE_H]),
                               rows=ex_show.round(3).to_dict("records"))

fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 0.32 * len(ex) + 1)), sharey=True)
yy = np.arange(len(ex))
axes[0].barh(yy, ex["raw_rate"] * 100, color=C_NORMAL, height=0.6)
axes[0].set_title("① 단순 해결률 (%)")
axes[0].set_yticks(yy, ex["tag"])
axes[1].barh(yy, ex["ratio"] * 100, color=[JUDGE_COLOR[j] for j in ex["judge"]], height=0.6)
axes[1].axvline(100, color="#52514e", lw=1, ls="--")
for i, (r, j, n) in enumerate(zip(ex["ratio"], ex["judge"], ex["n"])):
    axes[1].text(r * 100, i, f" {r*100:.0f}% ({n}문제) {j if j != '보통' else ''}", va="center", fontsize=9)
axes[1].set_title("② 레이팅 기대 대비 (%) · 판정")
fig.suptitle(f"실제 사용자 사례 {HANDLES[EXAMPLE_H]} (레이팅 약 {last_rating[EXAMPLE_H]:.0f}, 2024 상반기)", y=1.02)
savefig(fig, "그림4_사용자사례_태그진단")

# %% [markdown]
# ### 4-2. 난이도별 풀이 특성 (전체 사용자, 2024 상반기)
# 문제 난이도 구간마다 시도 수, 단순 해결률, 기대 해결률, 기대 대비를 계산합니다.
# 레이팅 구간과 겹쳐 보면 "자기 수준보다 어려운(쉬운) 문제를 레이팅에 비해 잘 푸는지"가 보입니다.
# 기대 확률은 태그 진단과 같은 값(하한·상한 5~95% 적용)을 씁니다. 상한 95% 때문에 거의 다 푸는 쉬운 문제·높은 레이팅에서는
# 기대가 실제보다 조금 낮게 잡혀 기대 대비가 100%를 약간 넘습니다. 그래서 절대값보다 구간끼리의 차이를 봅니다.

# %%
dgb = dg.assign(pc=np.clip(dg["pe"], *CONFIG["P_CLIP"]),
                dband=pd.cut(dg["difficulty"], CONFIG["DIFF_BANDS"], labels=CONFIG["DIFF_BAND_NAMES"], right=False),
                rband=pd.Categorical(BAND.reindex(dg["h"]).to_numpy(), categories=CONFIG["RATING_BAND_NAMES"], ordered=True))
diff_band = dgb.groupby("dband", observed=True).agg(시도수=("y", "size"), 사용자수=("h", "nunique"), 해결수=("y", "sum"),
                                                    기대해결수=("pc", "sum"), 첫제출정답률=("first_correct", "mean")).reset_index()
diff_band["시도 비중"] = diff_band["시도수"] / diff_band["시도수"].sum()
diff_band["단순 해결률"] = diff_band["해결수"] / diff_band["시도수"]
diff_band["기대 해결률"] = diff_band["기대해결수"] / diff_band["시도수"]
diff_band["기대 대비"] = diff_band["해결수"] / diff_band["기대해결수"]
diff_band = diff_band.rename(columns={"dband": "난이도 구간"})
show(diff_band.round(4))
TABLES["난이도 구간별 결과"] = diff_band

rd = dgb.groupby(["rband", "dband"], observed=True).agg(시도수=("y", "size"), 해결수=("y", "sum"), 기대해결수=("pc", "sum")).reset_index()
rd["단순 해결률"] = rd["해결수"] / rd["시도수"]
rd["기대 대비"] = rd["해결수"] / rd["기대해결수"]
rd = rd.rename(columns={"rband": "레이팅 구간", "dband": "난이도 구간"})
MIN_CELL = 300   # 시도가 이보다 적은 칸은 그림에서 '적음'으로 표시
TABLES["레이팅×난이도 결과"] = rd
RESULTS["difficulty"] = diff_band[["난이도 구간", "시도수", "단순 해결률", "기대 대비"]].round(4).astype({"난이도 구간": str}).to_dict("records")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
xb = np.arange(len(diff_band))
axes[0].bar(xb - 0.18, diff_band["단순 해결률"] * 100, width=0.34, color=C_OURS, label="단순 해결률")
axes[0].bar(xb + 0.18, diff_band["기대 해결률"] * 100, width=0.34, color=C_NORMAL, label="기대 해결률 (레이팅 기준)")
for i, r in diff_band.iterrows():
    axes[0].text(i, max(r["단순 해결률"], r["기대 해결률"]) * 100 + 1, f"기대 대비 {r['기대 대비']*100:.1f}%\n{r['시도수']:,}개",
                 ha="center", va="bottom", fontsize=8)
axes[0].set_xticks(xb, diff_band["난이도 구간"].astype(str))
axes[0].set_ylim(0, 115)
axes[0].set_ylabel("%")
axes[0].set_xlabel("문제 난이도 구간")
axes[0].set_title("난이도 구간별 단순 해결률과 기대 해결률")
axes[0].legend(frameon=False, loc="lower left", fontsize=9)
hm = rd.pivot(index="레이팅 구간", columns="난이도 구간", values="기대 대비")
hn = rd.pivot(index="레이팅 구간", columns="난이도 구간", values="시도수")
vals = np.where(hn.to_numpy() >= MIN_CELL, (hm.to_numpy() - 1) * 100, np.nan)
im = axes[1].imshow(vals, cmap="RdBu", vmin=-8, vmax=8, aspect="auto")
axes[1].set_xticks(range(hm.shape[1]), hm.columns.astype(str))
axes[1].set_yticks(range(hm.shape[0]), hm.index.astype(str))
axes[1].grid(False)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        n = hn.iat[i, j]
        txt = "적음" if not (n >= MIN_CELL) else f"{hm.iat[i, j]*100:.1f}%"
        axes[1].text(j, i, txt, ha="center", va="center", fontsize=8, color="#0b0b0b")
fig.colorbar(im, ax=axes[1], fraction=0.04, label="기대 대비 - 100 (%p)")
axes[1].set_xlabel("문제 난이도 구간")
axes[1].set_ylabel("사용자 레이팅 구간")
axes[1].set_title(f"레이팅 × 난이도 기대 대비 (시도 {MIN_CELL}개 미만은 '적음')")
savefig(fig, "그림9_난이도구간별_해결률과_기대대비")

# %% [markdown]
# ## 5. 검증 — 상반기에 진단한 약점이 하반기에도 기대보다 낮게 나오는가?
#
# - **기간 누수 방지**: 진단은 기준일(2024-07-01) 이전에 시작한 시도의 기준일 이전 제출만 사용
# - **관찰 부족**: 각 기간 끝 직전 7일 안에 시작한 시도 제외
# - **하반기 기대 확률**은 *하반기 당시의 레이팅*으로 다시 계산 → 하반기에 실력이 오른 것은 기대치에 반영됨
#
# **성공 기준**
# 1. 상반기 약점 태그의 하반기 (실제 − 기대)가 음수이고, 보통·강점보다 낮다 (판정 순서가 유지된다)
# 2. 사용자마다 같은 개수의 태그를 **단순 정답률 하위**로 고른 것보다 하반기에 더 확실히 기대 이하로 나온다
# 3. 하반기 시도 결과를 예측할 때 진단 점수를 더하면, 단순 정답률을 더한 것보다 예측이 더 잘 맞는다

# %%
def offset_logit_fit(off, X, y, lam=1e-3, iters=25):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = sigmoid(off + X @ b)
        H = (X * (mu * (1 - mu))[:, None]).T @ X + lam * np.eye(len(b))
        b += np.linalg.solve(H, X.T @ (y - mu) - lam * b)
    return b


def offset_logit_pred(off, X, b):
    return sigmoid(off + np.column_stack([np.ones(len(off)), X]) @ b)


def mean_ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - 1.96 * se, m + 1.96 * se


OURS = "레이팅 보정 진단 (z값)"


def validate(diag_tag, users, att_x, target, expect_fn, seed=CONFIG["SEED"], verbose=True):
    rng = np.random.default_rng(seed)
    dv = add_expectation(att_x[att_x["period"] == 2], target, expect_fn)
    dv = dv[dv["h"].isin(users)]
    vt = diagnose(dv, "tag")[["h", "tag", "n", "solved", "expected", "diff"]]
    vt.columns = ["h", "tag", "n_v", "solved_v", "expected_v", "diff_v"]
    pool = diag_tag[diag_tag["n"] >= CONFIG["MIN_TAG_ATT"]].merge(vt, on=["h", "tag"])
    pool = pool[pool["n_v"] >= CONFIG["MIN_VALID_ATT"]].copy()
    pool["resid_v"] = pool["diff_v"] / pool["n_v"]               # 하반기 시도당 (실제 − 기대)
    pool["below_v"] = (pool["solved_v"] < pool["expected_v"]).astype(int)
    pool["raw_rate_v"] = pool["solved_v"] / pool["n_v"]
    out = {"n_pairs": len(pool), "n_users": int(pool["h"].nunique())}

    # (1) 판정별 하반기 결과
    rows = []
    for col, mname in [("judge", OURS), ("judge_reg", "태그 회귀")]:
        if col not in pool:
            continue
        for j in ["약점", "보통", "강점"]:
            s = pool[pool[col] == j]
            if len(s) < 2:
                continue
            m, lo, hi = mean_ci(s["resid_v"])
            rows.append({"방법": mname, "상반기 판정": j, "쌍 수": len(s), "하반기 시도당 (실제−기대)": m,
                         "95% 하한": lo, "95% 상한": hi,
                         "하반기 기대 대비": s["solved_v"].sum() / s["expected_v"].sum(),
                         "하반기에도 기대 이하 비율": s["below_v"].mean()})
    by_judge = pd.DataFrame(rows)

    # (2) 같은 개수 비교: 사용자마다 우리 방법이 약점으로 고른 개수 k만큼 다른 방법으로도 고름
    pool["rnd"] = rng.random(len(pool))
    pool["k"] = pool.groupby("h")["judge"].transform(lambda s: (s == "약점").sum())
    sel = pool[pool["k"] > 0].copy()
    picks = {OURS: sel[sel["judge"] == "약점"]}
    for col, nm in [("raw_rate", "단순 정답률 하위"), ("z_reg", "태그 회귀"), ("rnd", "무작위 선택")]:
        if col in sel:
            r = sel.groupby("h")[col].rank(method="first")
            picks[nm] = sel[r <= sel["k"]]
    rows = []
    for name, s in picks.items():
        m, lo, hi = mean_ci(s["resid_v"])
        rows.append({"약점 고르는 방법": name, "고른 쌍 수": len(s), "하반기 시도당 (실제−기대)": m,
                     "95% 하한": lo, "95% 상한": hi, "하반기에도 기대 이하 비율": s["below_v"].mean(),
                     "하반기 단순 해결률": s["solved_v"].sum() / s["n_v"].sum()})
    weak_cmp = pd.DataFrame(rows)

    # (3) 사용자 안 순위 상관: 상반기 점수 순서가 하반기 순서와 맞는가 (태그 5개 이상 사용자)
    pm = pool[pool.groupby("h")["tag"].transform("size") >= 5]
    rank_rows = []
    for xcol, nm in [("score", "보정 점수(평활화)"), ("z", "z값"), ("beta", "태그 회귀 계수"), ("raw_rate", "단순 정답률")]:
        if xcol not in pm:
            continue
        rx = pm.groupby("h")[xcol].rank()
        for ycol, ynm in [("resid_v", "하반기 (실제−기대)"), ("raw_rate_v", "하반기 단순 해결률")]:
            ry = pm.groupby("h")[ycol].rank()
            r = pd.DataFrame({"h": pm["h"], "x": rx, "y": ry}).groupby("h").apply(lambda s: s["x"].corr(s["y"]))
            rank_rows.append({"상반기 지표": nm, "하반기 기준": ynm, "사용자 평균 순위상관": r.mean(), "사용자 수": len(r)})
    rank_cmp = pd.DataFrame(rank_rows)

    # (4) 시도 단위 예측: 하반기 시도마다 logit = logit(p_기대) + a + b·특징 (특징은 모두 상반기 값)
    #     사용자 절반으로 a,b를 맞추고 나머지 절반으로 평가
    long = tag_long(dv)
    fcols = ["score", "raw_rate"] + (["beta"] if "beta" in diag_tag else [])
    long = long.merge(diag_tag[["h", "tag"] + fcols], on=["h", "tag"], how="left")
    user_raw = diag_tag.groupby("h")[["solved", "n"]].sum().eval("solved / n")
    long["raw_rate"] = long["raw_rate"].fillna(long["h"].map(user_raw))   # 상반기에 안 푼 태그는 그 사용자 전체 해결률
    long = long.fillna({"score": 0, "beta": 0})
    A = dv.join(long.groupby("aid")[fcols].mean(), how="inner")
    users_v = A["h"].unique()
    te = A["h"].isin(set(rng.choice(users_v, size=len(users_v) // 2, replace=False))).to_numpy()
    pcl = np.clip(A["pe"].to_numpy(), *CONFIG["P_CLIP"])
    off = np.log(pcl / (1 - pcl))
    y = A["y"].to_numpy()
    pred_rows = []
    specs = [("① 기대 확률만", []), ("② ① + 단순 정답률", ["raw_rate"]), ("③ ① + 보정 점수 (우리)", ["score"])]
    if "beta" in A:
        specs.append(("④ ① + 태그 회귀 계수", ["beta"]))
    specs.append(("⑤ ① + 보정 점수 + 단순 정답률", ["score", "raw_rate"]))
    for nm, cols in specs:
        X = A[cols].to_numpy() if cols else np.zeros((len(A), 0))
        b = offset_logit_fit(off[~te], X[~te], y[~te])
        pred_rows.append({"모델": nm, **eval_probs(y[te], offset_logit_pred(off[te], X[te], b))})
    pred_cmp = pd.DataFrame(pred_rows)
    pred_cmp["①대비 log-loss 개선(%)"] = (pred_cmp.loc[0, "logloss"] - pred_cmp["logloss"]) / pred_cmp.loc[0, "logloss"] * 100

    out.update(by_judge=by_judge, weak_cmp=weak_cmp, rank_cmp=rank_cmp, pred_cmp=pred_cmp, pool=pool)
    if verbose:
        print(f"검증 쌍 {out['n_pairs']:,}개 (사용자 {out['n_users']:,}명)")
        print("\n[1] 상반기 판정별 하반기 결과\n", by_judge.round(4).to_string(index=False))
        print("\n[2] 같은 개수로 약점 고르기 비교\n", weak_cmp.round(4).to_string(index=False))
        print("\n[3] 사용자 안 순위 상관\n", rank_cmp.round(4).to_string(index=False))
        print("\n[4] 하반기 시도 예측 (테스트 사용자)\n", pred_cmp.round(5).to_string(index=False))
    return out


t0 = time.time()
VAL = validate(diag_tag, DIAG_USERS, att, TARGET, expect)
print(f"({time.time()-t0:.1f}초)")
TABLES["표2a 판정별 하반기 결과"] = VAL["by_judge"]
TABLES["표2b 약점 선택 방법 비교"] = VAL["weak_cmp"]
TABLES["표2c 순위 상관"] = VAL["rank_cmp"]
TABLES["표2d 시도 예측 비교"] = VAL["pred_cmp"]
RESULTS["validation"] = {k: VAL[k].round(4).to_dict("records") for k in ["by_judge", "weak_cmp", "rank_cmp", "pred_cmp"]}

# 실제 사용자 사례의 상반기 판정이 하반기에도 이어졌는지 (위와 같은 검증 기준: 하반기 레이팅으로 다시 계산한 기대 대비)
ex_v = VAL["pool"][VAL["pool"]["h"] == EXAMPLE_H].copy()
ex_v["ratio_v"] = ex_v["solved_v"] / ex_v["expected_v"]
ex_v = ex_v.sort_values("ratio")[["tag", "n", "ratio", "judge", "n_v", "solved_v", "ratio_v"]].rename(columns={
    "tag": "태그", "n": "상반기 시도", "ratio": "상반기 기대 대비", "judge": "상반기 판정",
    "n_v": "하반기 시도", "solved_v": "하반기 해결", "ratio_v": "하반기 기대 대비"})
print(f"\n실제 사용자 사례 {HANDLES[EXAMPLE_H]}의 하반기 확인 (하반기 시도 {CONFIG['MIN_VALID_ATT']}개 이상 태그)")
print(ex_v.round(3).to_string(index=False))
TABLES["사례 사용자 하반기 확인"] = ex_v
RESULTS["example_user_h2"] = ex_v.round(4).to_dict("records")

# %%
# 그림 5: 검증 결과
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
bj = VAL["by_judge"][VAL["by_judge"]["방법"] == OURS]
x = np.arange(len(bj))
mid = bj["하반기 시도당 (실제−기대)"] * 100
axes[0].bar(x, mid, color=[JUDGE_COLOR[j] for j in bj["상반기 판정"]], width=0.55)
axes[0].errorbar(x, mid, yerr=[mid - bj["95% 하한"] * 100, bj["95% 상한"] * 100 - mid], fmt="none", ecolor="#52514e", capsize=4, lw=1)
for i, v in enumerate(mid):
    axes[0].text(i, v, f"{v:+.2f}%p", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
axes[0].set_xticks(x, [f"상반기 {j}\n({n:,}쌍)" for j, n in zip(bj["상반기 판정"], bj["쌍 수"])])
axes[0].axhline(0, color="#52514e", lw=0.8)
axes[0].set_ylabel("하반기 시도당 (실제 - 기대), %p")
axes[0].set_title("상반기 판정별 하반기 결과 (95% 신뢰구간)")
wc = VAL["weak_cmp"]
col_map = {OURS: C_OURS, "단순 정답률 하위": C_RAW, "태그 회귀": C_EXTRA, "무작위 선택": C_RAND}
x = np.arange(len(wc))
axes[1].bar(x, wc["하반기에도 기대 이하 비율"] * 100, color=[col_map.get(n, C_RAND) for n in wc["약점 고르는 방법"]], width=0.55)
for i, v in enumerate(wc["하반기에도 기대 이하 비율"]):
    axes[1].text(i, v * 100, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
axes[1].set_xticks(x, [s.replace(" (", "\n(") for s in wc["약점 고르는 방법"]], fontsize=9)
axes[1].set_ylabel("하반기에도 기대 이하인 비율 (%)")
axes[1].set_title("같은 개수로 약점을 골랐을 때 하반기 적중률")
lo = max(0, wc["하반기에도 기대 이하 비율"].min() * 100 - 15)
axes[1].set_ylim(lo, min(100, wc["하반기에도 기대 이하 비율"].max() * 100 + 10))
savefig(fig, "그림5_검증결과")

# %% [markdown]
# ## 6. 비교 실험
# ### 6-1. 정답값 기준: 최종 해결(기본) vs 첫 제출 정답
# 같은 파이프라인을 정답값만 바꿔 다시 돌려, 기대 확률 모델 성능과 진단의 하반기 재현성을 비교합니다.
# - 최종 해결: 결국 풀어냈는지 → 실력에 가장 가까움 (기본)
# - 첫 제출 정답: 한 번에 맞혔는지 → 신중함(행동)과 실력이 섞임

# %%
def run_pipeline(att_x, target, expect_fn=None):
    if expect_fn is None:
        ms, _ = fit_expectation_models(att_x, target)
        d1 = att_x[(att_x["period"] == 1) & (att_x["in_model"] == 1)]
        expect_fn = min(ms.values(), key=lambda f: log_loss(d1[target], np.clip(f(d1), 1e-6, 1 - 1e-6)))
    d_g, dtag, _ = run_diagnosis(att_x, target, expect_fn)
    val = validate(dtag, d_g["h"].unique(), att_x, target, expect_fn, verbose=False)
    bj = val["by_judge"][val["by_judge"]["방법"] == OURS].set_index("상반기 판정")
    wcmp = val["weak_cmp"].set_index("약점 고르는 방법")
    pc = val["pred_cmp"].set_index("모델")
    rk = val["rank_cmp"]
    rk = rk[rk["하반기 기준"] == "하반기 (실제−기대)"].set_index("상반기 지표")["사용자 평균 순위상관"]
    d_v = add_expectation(att_x[att_x["period"] == 2], target, expect_fn)
    return {
        "진단 사용자 수": d_g["h"].nunique(),
        "진단 기간 시도 수": len(d_g),
        "정답값 평균": d_g["y"].mean(),
        "기대모델 AUC(검증)": roc_auc_score(d_v["y"], d_v["pe"]),
        "약점 비율": (dtag["judge"] == "약점").mean(),
        "검증 쌍 수": val["n_pairs"],
        "약점→하반기 시도당(실제−기대)": bj.loc["약점", "하반기 시도당 (실제−기대)"],
        "강점→하반기 시도당(실제−기대)": bj.loc["강점", "하반기 시도당 (실제−기대)"],
        "약점 적중률(우리)": wcmp.loc[OURS, "하반기에도 기대 이하 비율"],
        "약점 적중률(단순 정답률)": wcmp.loc["단순 정답률 하위", "하반기에도 기대 이하 비율"],
        "약점 적중률(무작위)": wcmp.loc["무작위 선택", "하반기에도 기대 이하 비율"],
        "순위상관(우리)": rk.loc["보정 점수(평활화)"],
        "순위상관(단순 정답률)": rk.loc["단순 정답률"],
        "예측 개선%(우리)": pc.loc["③ ① + 보정 점수 (우리)", "①대비 log-loss 개선(%)"],
        "예측 개선%(단순 정답률)": pc.loc["② ① + 단순 정답률", "①대비 log-loss 개선(%)"],
    }


t0 = time.time()
target_cmp = pd.DataFrame({tg: run_pipeline(att, tg) for tg in ["solved", "first_correct"]}).T
target_cmp.index = ["최종 해결 (solved, 기본)", "첫 제출 정답 (first_correct)"]
print(target_cmp.T.to_string(float_format=lambda v: f"{v:,.4f}"))
print(f"({time.time()-t0:.1f}초)")
TABLES["표3 정답값 기준 비교"] = target_cmp.reset_index().rename(columns={"index": "정답값"})
RESULTS["target_cmp"] = target_cmp.round(4).to_dict("index")

# %% [markdown]
# ### 6-2. 기간 배정 규칙: 첫 제출 기준 vs 마지막 제출 기준 vs 기준일에서 자르기
#
# | 규칙 | 시도의 기간 | 문제점 |
# |---|---|---|
# | **첫 제출 (기본)** | 첫 제출이 속한 기간 | 진단 기간에 시작해 기준일 이후 푼 문제는 진단에서 '미해결'로 남음 (관찰 부족 7일 제외로 완화) |
# | 마지막 제출 | 마지막 제출이 속한 기간 | 기준일 시점에는 알 수 없는 미래 정보로 진단 집합이 바뀜(누수) |
# | 기준일에서 자르기 | 제출마다 그 기간 | 하반기에 '이미 시도했던 문제'가 새 시도로 섞임 |
#
# 같은 기대 확률 모델로 세 규칙을 돌려 시도 수·해결률·검증 성능을 비교합니다.

# %%
t0 = time.time()
rule_rows = {}
first_seen = sub.groupby(["h", "p"])["t"].min()
for rule in ["first", "last", "split"]:
    a_r = att if rule == CONFIG["PERIOD_RULE"] else build_attempts(sub, rule)[0]
    r = run_pipeline(a_r, TARGET, expect_fn=expect)
    d1 = a_r[a_r["period"] == 1]
    d2 = a_r[a_r["period"] == 2]
    fs1 = first_seen.reindex(pd.MultiIndex.from_arrays([d1["h"], d1["p"]])).to_numpy()
    fs2 = first_seen.reindex(pd.MultiIndex.from_arrays([d2["h"], d2["p"]])).to_numpy()
    rule_rows[rule] = {
        "진단 기간 시도 수": len(d1),
        "진단 기간 최종 해결률": d1["solved"].mean(),
        "진단 시도 중 진단 기간 이전에 시작한 비율": (fs1 < T_DIAG).mean(),
        "검증 시도 중 기준일 이전에 이미 시도한 비율": (fs2 < T_CUT).mean(),
        **r}
rule_cmp = pd.DataFrame(rule_rows).T
rule_cmp.index = ["첫 제출 기준 (기본)", "마지막 제출 기준", "기준일에서 자르기"]
print(rule_cmp.T.to_string(float_format=lambda v: f"{v:,.4f}"))
print(f"({time.time()-t0:.1f}초)")
TABLES["표4 기간 배정 규칙 비교"] = rule_cmp.reset_index().rename(columns={"index": "규칙"})
RESULTS["rule_cmp"] = rule_cmp.round(4).to_dict("index")

# %% [markdown]
# ## 7. 행동 지표와 학습 유형 군집화
#
# 진단 기간의 행동 지표를 사용자마다 계산합니다. 그대로 군집화하면 레이팅 순으로만 나뉘므로
# **레이팅 구간 안에서 표준화(z-score)** 한 뒤 K-means를 적용합니다.

# %%
def behavior_metrics(att_p, gaps_p, sub_p, users):
    a = att_p[(att_p["short_obs"] == 0) & att_p["h"].isin(users)]
    g = a.groupby("h")
    m = pd.DataFrame({
        "시도 문제 수": g.size(),
        "해결률": g["solved"].mean(),
        "포기율": 1 - g["solved"].mean(),
        "첫 제출 정답률": g["first_correct"].sum() / g["solved"].sum().clip(lower=1),
        "평균 재시도": a[a["solved"] == 1].groupby("h")["retries"].mean(),
    })
    err = g[["wa", "tle", "re", "ce", "other"]].sum()
    tot = err.sum(axis=1).clip(lower=1)
    m["WA 비율"], m["TLE·MLE 비율"], m["RE 비율"], m["CE 비율"], m["기타 오답 비율"] = [err[c] / tot for c in ["wa", "tle", "re", "ce", "other"]]
    m["도전 성향"] = a[a["in_model"] == 1].groupby("h")["gap"].mean()
    tl = a[["h", "p"]].merge(PTAGS, on="p").groupby(["h", "tag"]).size()
    def norm_entropy(s):
        q = s / s.sum()
        return float(-(q * np.log(q)).sum() / np.log(len(ALL_TAGS)))
    m["태그 다양성"] = tl.groupby("h").apply(norm_entropy)
    m["재제출 간격 중앙값(분)"] = gaps_p[gaps_p["h"].isin(users)].groupby("h")["gap_min"].median()
    wk = (sub_p["t"] // (7 * 86400))
    w = pd.DataFrame({"h": sub_p["h"], "wk": wk}).drop_duplicates()
    span = w.groupby("h")["wk"].agg(lambda s: s.max() - s.min() + 1)
    m["활동 주 수"] = w.groupby("h").size()
    m["꾸준함"] = m["활동 주 수"] / span
    m["주당 시도 수"] = m["시도 문제 수"] / m["활동 주 수"]
    return m


t0 = time.time()
sub1 = sub[sub["per"] == 1]
beh = behavior_metrics(att[att["period"] == 1], gaps[gaps["period"] == 1], sub1, DIAG_USERS)
ratio_user = dg.groupby("h")[["y", "pe"]].sum().eval("y / pe")
beh["전체 기대 대비"] = ratio_user          # 레이팅 기대 대비 전체 수준
beh["최근 레이팅"] = last_rating
beh["레이팅 구간"] = BAND
# 하반기 레이팅 변화 (군집 해석용)
lr2 = sub[(sub["per"] == 2)].groupby("h")["rating"].last()
beh["하반기 레이팅 변화"] = lr2.reindex(beh.index) - beh["최근 레이팅"]
beh = beh.dropna(subset=["평균 재시도", "재제출 간격 중앙값(분)", "도전 성향"])
print(f"행동 지표 사용자 {len(beh):,}명 ({time.time()-t0:.1f}초)")
print(beh.describe().T.round(3).to_string())
TABLES["행동 지표 기술통계"] = beh.describe().T.reset_index()

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

# 군집화에 쓰는 지표 (제출 정답률은 평균 재시도와 거의 같은 정보라 제외, 성과 지표인 기대 대비도 제외)
CLUSTER_FEATURES = ["포기율", "첫 제출 정답률", "평균 재시도", "WA 비율", "TLE·MLE 비율", "RE 비율", "CE 비율",
                    "도전 성향", "태그 다양성", "재제출 간격 중앙값(분)", "꾸준함", "주당 시도 수"]
LOG_FEATURES = ["평균 재시도", "재제출 간격 중앙값(분)", "주당 시도 수"]

X0 = beh[CLUSTER_FEATURES].copy()
for c in LOG_FEATURES:
    X0[c] = np.log1p(X0[c])


def standardize(X, groups=None):
    if groups is None:
        Z = (X - X.mean()) / X.std()
    else:
        Z = X.groupby(groups, observed=True).transform(lambda s: (s - s.mean()) / s.std())
    return Z.clip(-3, 3).fillna(0)


Z_band = standardize(X0, beh["레이팅 구간"])     # 레이팅 구간 안 표준화 (기본)
Z_glob = standardize(X0)                         # 비교용: 전체 표준화

rng = np.random.default_rng(CONFIG["SEED"])
sil_idx = rng.choice(len(Z_band), size=min(6000, len(Z_band)), replace=False)
k_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=CONFIG["SEED"]).fit(Z_band)
    k_rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z_band.iloc[sil_idx], km.labels_[sil_idx])})
k_sel = pd.DataFrame(k_rows)
TABLES["군집 수 선택"] = k_sel
print(k_sel.round(4).to_string(index=False))
# 해석 가능성을 위해 4~6개 중 실루엣이 가장 높은 k (필요하면 K_FINAL을 직접 지정)
K_FINAL = int(k_sel[k_sel["k"].between(4, 6)].sort_values("silhouette").iloc[-1]["k"])
print("선택한 군집 수:", K_FINAL)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(k_sel["k"], k_sel["inertia"], marker="o", color=C_OURS, lw=2)
axes[0].set_title("엘보우 (군집 내 제곱합)")
axes[1].plot(k_sel["k"], k_sel["silhouette"], marker="o", color=C_OURS, lw=2)
axes[1].axvline(K_FINAL, color=C_RAW, ls="--", lw=1)
axes[1].set_title("실루엣 계수")
for ax in axes:
    ax.set_xlabel("군집 수 k")
savefig(fig, "그림6_군집수_선택")

# %%
km = KMeans(n_clusters=K_FINAL, n_init=20, random_state=CONFIG["SEED"]).fit(Z_band)
beh["군집"] = km.labels_
centers = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)

# 유형 이름 규칙: 군집 중심(레이팅 구간 내 z)에서 절댓값이 가장 큰 지표 2개를 골라,
# 그 지표가 평균보다 높으면 앞 단어, 낮으면 뒤 단어를 붙인다. (사람이 지은 이름이 아니라 데이터에서 정해지는 이름)
FEATURE_WORDS = {
    "포기율": ("쉽게 포기", "끝까지 해결"), "첫 제출 정답률": ("신중", "일단 제출"), "평균 재시도": ("시행착오", "재시도 적음"),
    "WA 비율": ("로직 오류 많음", "로직 오류 적음"), "TLE·MLE 비율": ("효율 부족", "효율 양호"), "RE 비율": ("예외 처리 약함", "예외 처리 양호"),
    "CE 비율": ("문법·환경 실수", "문법 실수 적음"), "도전 성향": ("도전", "안전"), "태그 다양성": ("균형", "편식"),
    "재제출 간격 중앙값(분)": ("숙고", "연속 제출"), "꾸준함": ("꾸준", "몰아치기"), "주당 시도 수": ("다작", "소량"),
}
names = {}
for c in range(K_FINAL):
    top = centers.loc[c].abs().sort_values(ascending=False).index[:2]
    names[c] = "·".join(FEATURE_WORDS[f][0 if centers.loc[c, f] > 0 else 1] for f in top) + "형"
AUTO_NAMES = dict(names)   # 규칙으로 정해진 이름 (근거 확인용으로 결과표에 남김)

# 팀이 정한 자연스러운 이름: 위 자동 이름(가장 두드러진 지표 2개)과 그림7 히트맵의 표준점수를 보고 정함
TYPE_NAME_MAP = {
    "다작·꾸준형": "꾸준한 다작형",              # 주당 시도 수↑, 꾸준함↑, 포기율↓
    "편식·로직 오류 많음형": "안전 지향형",       # 도전 성향↓, 태그 다양성↓, WA 비율↑ → 쉬운 문제·익숙한 유형 위주
    "시행착오·일단 제출형": "도전적 시행착오형",  # 도전 성향↑, 평균 재시도↑, 첫 제출 정답률↓
    "로직 오류 적음·효율 부족형": "효율 개선 필요형",  # TLE·MLE 비율↑, WA 비율↓ → 로직은 맞는데 시간·메모리 초과
}
TYPE_DESC = {
    "꾸준한 다작형": "매주 꾸준히, 많은 문제를 푸는 유형",
    "안전 지향형": "자기 수준보다 쉬운 문제와 익숙한 유형 위주로 풀고, 틀릴 때는 로직 오류(WA)가 많은 유형",
    "도전적 시행착오형": "자기 수준보다 어려운 문제에 도전하고, 여러 번 고쳐 제출하며 맞히는 유형",
    "효율 개선 필요형": "로직 오류는 적지만 시간·메모리 초과(TLE·MLE)가 많아 효율을 개선해야 하는 유형",
}
names = {c: TYPE_NAME_MAP.get(n, n) for c, n in AUTO_NAMES.items()}   # 표에 없는 이름은 자동 이름 그대로
print("유형 이름:", {AUTO_NAMES[c]: names[c] for c in names})
beh["유형"] = beh["군집"].map(names)

profile = beh.groupby("유형")[CLUSTER_FEATURES + ["전체 기대 대비", "최근 레이팅", "하반기 레이팅 변화"]].mean()
profile.insert(0, "사용자 수", beh["유형"].value_counts())
profile.insert(1, "자동 이름(규칙)", profile.index.map({v: AUTO_NAMES[k] for k, v in names.items()}))
profile.insert(2, "설명", profile.index.map(TYPE_DESC))
print(profile.round(3).T.to_string())
TABLES["표5 학습 유형 프로필"] = profile.reset_index()

# 학습 유형별 난이도 분포와 기대 대비 (2024 상반기 시도)
dgt = dgb.assign(유형=beh["유형"].reindex(dgb["h"]).to_numpy()).dropna(subset=["유형"])
type_diff = dgt.groupby(["유형", "dband"], observed=True).agg(시도수=("y", "size"), 해결수=("y", "sum"), 기대해결수=("pc", "sum")).reset_index()
type_diff["시도 비중"] = type_diff["시도수"] / type_diff.groupby("유형")["시도수"].transform("sum")
type_diff["기대 대비"] = type_diff["해결수"] / type_diff["기대해결수"]
type_diff = type_diff.rename(columns={"dband": "난이도 구간"})
print(type_diff.pivot(index="유형", columns="난이도 구간", values="시도 비중").round(3).to_string())
TABLES["학습 유형×난이도 결과"] = type_diff
TABLES["군집 중심(구간 내 z)"] = centers.rename(index=names).reset_index().rename(columns={"index": "유형"})


def cramers_v(a, b):
    ct = pd.crosstab(a, b).to_numpy()
    n = ct.sum()
    exp = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = ((ct - exp) ** 2 / exp).sum()
    return math.sqrt(chi2 / (n * (min(ct.shape) - 1)))


km_g = KMeans(n_clusters=K_FINAL, n_init=20, random_state=CONFIG["SEED"]).fit(Z_glob)
cv_band = cramers_v(beh["군집"], beh["레이팅 구간"])
cv_glob = cramers_v(km_g.labels_, beh["레이팅 구간"])
print(f"군집-레이팅 구간 연관도(Cramér's V): 구간 내 표준화 {cv_band:.3f} vs 전체 표준화 {cv_glob:.3f} (낮을수록 레이팅과 독립)")
# 안정성: 80% 표본으로 5번 다시 군집화해 원래 결과와 일치도(ARI)
aris = []
for s in range(5):
    idx = np.random.default_rng(s).choice(len(Z_band), size=int(0.8 * len(Z_band)), replace=False)
    lab = KMeans(n_clusters=K_FINAL, n_init=10, random_state=s).fit(Z_band.iloc[idx]).predict(Z_band)
    aris.append(adjusted_rand_score(km.labels_, lab))
print(f"군집 안정성 ARI 평균 {np.mean(aris):.3f} (1에 가까울수록 안정)")
RESULTS["cluster"] = dict(k=K_FINAL, names=names, sizes=beh["유형"].value_counts().to_dict(),
                          cramers_v_band=cv_band, cramers_v_global=cv_glob, ari_mean=float(np.mean(aris)))

# %%
# 그림 7: 유형 프로필 히트맵 + 레이팅 구간 구성
fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * K_FINAL + 3), gridspec_kw={"width_ratios": [3, 2]})
cm = centers.rename(index=names)
im = axes[0].imshow(cm.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
axes[0].set_xticks(range(len(CLUSTER_FEATURES)), CLUSTER_FEATURES, rotation=45, ha="right", fontsize=9)
axes[0].set_yticks(range(K_FINAL), [f"{n} ({(beh['군집']==i).sum():,}명)" for i, n in names.items()], fontsize=9)
axes[0].grid(False)
for i in range(K_FINAL):
    for j in range(len(CLUSTER_FEATURES)):
        axes[0].text(j, i, f"{cm.iat[i, j]:.1f}", ha="center", va="center", fontsize=7, color="#0b0b0b")
fig.colorbar(im, ax=axes[0], fraction=0.03, label="레이팅 구간 내 z")
axes[0].set_title("학습 유형별 행동 지표 (구간 내 표준화)")
comp = pd.crosstab(beh["유형"], beh["레이팅 구간"], normalize="columns")
bottom = np.zeros(comp.shape[1])
palette = [C_OURS, C_RAW, C_EXTRA, "#eda100", "#e87ba4", "#6250d6"]
for i, (nm, row) in enumerate(comp.iterrows()):
    axes[1].bar(comp.columns.astype(str), row.values, bottom=bottom, color=palette[i % len(palette)], label=nm, width=0.7, edgecolor="white", linewidth=1)
    bottom += row.values
axes[1].set_title(f"레이팅 구간별 유형 구성 (Cramér's V={cv_band:.2f})")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1, 1), loc="upper left")
savefig(fig, "그림7_학습유형_프로필")

pca = PCA(n_components=2, random_state=0).fit(Z_band)
P2 = pca.transform(Z_band)
fig, ax = plt.subplots(figsize=(6.5, 5))
for i, nm in names.items():
    s = beh["군집"].to_numpy() == i
    ax.scatter(P2[s, 0], P2[s, 1], s=4, alpha=0.35, color=palette[i % len(palette)], label=nm)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("학습 유형 분포 (PCA 2차원)")
ax.legend(frameon=False, markerscale=4, fontsize=8)
savefig(fig, "그림8_학습유형_분포")

# 그림 10: 학습 유형별로 어떤 난이도의 문제를 많이 시도하는지
share = type_diff.pivot(index="유형", columns="난이도 구간", values="시도 비중")
share = share.loc[beh["유형"].value_counts().index]
fig, ax = plt.subplots(figsize=(11, 0.6 * len(share) + 1.8))
left = np.zeros(len(share))
dcols = ["#cfe0f5", "#9cc1ea", "#5f9be0", "#2a78d6", "#18518f"]
for k, c in enumerate(share.columns):
    v = share[c].fillna(0).to_numpy()
    ax.barh(range(len(share)), v * 100, left=left * 100, color=dcols[k % len(dcols)], height=0.6, label=str(c), edgecolor="white", linewidth=1)
    for i in range(len(share)):
        if v[i] >= 0.06:
            ax.text((left[i] + v[i] / 2) * 100, i, f"{v[i]*100:.0f}%", ha="center", va="center", fontsize=8,
                    color="#0b0b0b" if k < 2 else "white")
    left += v
ax.set_yticks(range(len(share)), share.index)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("시도 비중 (%)")
ax.set_title("학습 유형별 시도한 문제의 난이도 분포 (2024 상반기)")
ax.legend(title="난이도 구간", frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
savefig(fig, "그림10_학습유형별_난이도분포")

# %% [markdown]
# ## 8. 웹·LLM 연결용 사용자 진단 내보내기
# 사용자마다 진단 결과를 JSON 한 줄로 저장합니다. 웹 화면은 이 파일을 읽어 진단을 보여 주고,
# LLM에는 `build_llm_prompt()`가 만든 문장을 시스템 프롬프트로 넣어 진단을 설명하게 합니다.

# %%
pct = beh[CLUSTER_FEATURES].groupby(beh["레이팅 구간"], observed=True).rank(pct=True)
diag_cat_ok = diag_cat[diag_cat["n"] >= CONFIG["MIN_TAG_ATT"]]
dband = dg.assign(dband=pd.cut(dg["difficulty"], CONFIG["DIFF_BANDS"], labels=CONFIG["DIFF_BAND_NAMES"], right=False)) \
          .groupby(["h", "dband"], observed=True).agg(n=("y", "size"), solved=("y", "sum"), expected=("pe", "sum")).reset_index()


def user_profile(h):
    h = int(h)
    t = diag_tag[(diag_tag["h"] == h) & (diag_tag["n"] >= CONFIG["MIN_TAG_ATT"])]
    t = t[~t["tag"].isin(LOW_DISCRIM_TAGS)]
    def rows(df):
        return [dict(tag=r.tag, category=TAG_CATEGORY.get(r.tag, "기타"), attempts=int(r.n), correct=int(r.solved),
                     raw_rate=round(r.raw_rate, 3), vs_expected=round(r.ratio, 3), z=round(r.z, 2)) for r in df.itertuples()]
    b = beh.loc[h] if h in beh.index else None
    return {
        "handle": str(HANDLES[h]),
        "rating": float(last_rating.get(h, np.nan)),
        "rating_band": str(BAND.get(h)),
        "learning_type": names.get(int(b["군집"])) if b is not None else None,
        "learning_type_desc": TYPE_DESC.get(names.get(int(b["군집"]))) if b is not None else None,
        "target": TARGET,
        "overall_vs_expected": round(float(ratio_user.get(h, np.nan)), 3),
        "weak_tags": rows(t[t["judge"] == "약점"].sort_values("z")),
        "strong_tags": rows(t[t["judge"] == "강점"].sort_values("z", ascending=False)),
        "categories": [dict(category=r.category, attempts=int(r.n), vs_expected=round(r.ratio, 3), judge=r.judge)
                       for r in diag_cat_ok[diag_cat_ok["h"] == h].sort_values("z").itertuples()],
        "difficulty_bands": [dict(band=str(r.dband), attempts=int(r.n), solve_rate=round(r.solved / r.n, 3),
                                  vs_expected=round(r.solved / r.expected, 3)) for r in dband[dband["h"] == h].itertuples()],
        "behavior": None if b is None else {k: (round(float(b[k]), 3)) for k in CLUSTER_FEATURES},
        "behavior_percentile_in_band": None if h not in pct.index else {k: round(float(pct.loc[h, k]), 2) for k in CLUSTER_FEATURES},
    }


def build_llm_prompt(profile):
    weak = ", ".join(f"{w['tag']}(기대 대비 {w['vs_expected']*100:.0f}%, {w['attempts']}문제)" for w in profile["weak_tags"][:3]) or "뚜렷한 약점 없음"
    strong = ", ".join(f"{s['tag']}(기대 대비 {s['vs_expected']*100:.0f}%, {s['attempts']}문제)" for s in profile["strong_tags"][:3]) or "뚜렷한 강점 없음"
    return (
        "너는 알고리즘 문제 풀이 학습 코치다. 아래 진단은 학습자의 Codeforces 제출 이력을 레이팅 대비로 분석한 결과다.\n"
        f"- 레이팅: {profile['rating']:.0f} ({profile['rating_band']} 구간)\n"
        f"- 레이팅 대비 전체 수준: 기대 해결 수의 {profile['overall_vs_expected']*100:.0f}%\n"
        f"- 학습 유형: {profile['learning_type']}\n"
        f"- 레이팅 대비 약한 유형: {weak}\n"
        f"- 레이팅 대비 강한 유형: {strong}\n"
        "규칙: (1) '기대 대비'는 같은 레이팅의 학습자가 같은 난이도 문제를 풀 것으로 기대되는 수 대비 실제로 푼 비율임을 쉽게 설명한다. "
        "(2) 코드나 정답을 바로 주지 말고 개념 설명 → 접근 힌트 → 단계별 힌트 순으로 수준에 맞게 안내한다. "
        "(3) 진단은 제출 기록만으로 만든 것이므로 틀린 원인은 사용자가 준 코드·질문을 보고 판단한다."
    )


t0 = time.time()
with open(OUT / "웹LLM용_사용자별_진단.jsonl", "w", encoding="utf-8") as f:
    for h in beh.index:
        f.write(json.dumps(user_profile(h), ensure_ascii=False) + "\n")
ex_profile = user_profile(EXAMPLE_H)
(OUT / "웹LLM용_사례사용자_진단.json").write_text(json.dumps(ex_profile, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "웹LLM용_사례사용자_프롬프트.txt").write_text(build_llm_prompt(ex_profile), encoding="utf-8")
print(f"웹LLM용_사용자별_진단.jsonl {len(beh):,}명 ({time.time()-t0:.1f}초)")
print(build_llm_prompt(ex_profile))

# 전체 결과표 CSV (웹 DB 적재용)
diag_tag.assign(handle=HANDLES[diag_tag["h"]]).drop(columns="h").to_csv(OUT / "진단_사용자별_태그.csv", index=False, encoding="utf-8-sig")
diag_cat.assign(handle=HANDLES[diag_cat["h"]]).drop(columns="h").to_csv(OUT / "진단_사용자별_상위범주.csv", index=False, encoding="utf-8-sig")
beh.assign(handle=HANDLES[beh.index]).to_csv(OUT / "사용자별_행동지표와_학습유형.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# ## 9. 결과 요약 (발표·책자용)
# 아래 셀이 핵심 숫자를 `결과요약.md`와 `결과표.xlsx`로 저장합니다. 숫자는 이 파일에서 그대로 옮겨 쓰세요.

# %%
def fmt_pct(x):
    return f"{x*100:.1f}%"


bj = VAL["by_judge"][VAL["by_judge"]["방법"] == OURS].set_index("상반기 판정")
wc = VAL["weak_cmp"].set_index("약점 고르는 방법")
pc = VAL["pred_cmp"].set_index("모델")
lines = [
    "# 분석 결과 요약 (자동 생성)", "",
    f"- 제출 {RESULTS['n_submissions']:,}건, 사용자 {RESULTS['n_users']:,}명, 공식 문제 병합률 {fmt_pct(RESULTS['merge_rate'])}",
    f"- 진단 대상: 2024 상반기 시도 {CONFIG['MIN_USER_ATT']}개 이상 사용자 {RESULTS['n_diag_users']:,}명, 시도 {RESULTS['n_diag_attempts']:,}개",
    f"- 기대 확률 모델: **{RESULTS['expect_model']}** (진단 기간 log-loss로 선택)",
    "", "## 검증 (2024 상반기 진단 → 하반기 확인)",
    f"- 상반기 **약점** 태그의 하반기 시도당 (실제−기대): {bj.loc['약점','하반기 시도당 (실제−기대)']*100:+.2f}%p "
    f"(95% CI {bj.loc['약점','95% 하한']*100:+.2f} ~ {bj.loc['약점','95% 상한']*100:+.2f}), 하반기에도 기대 이하 {fmt_pct(bj.loc['약점','하반기에도 기대 이하 비율'])}",
    f"- 보통: {bj.loc['보통','하반기 시도당 (실제−기대)']*100:+.2f}%p / 강점: {bj.loc['강점','하반기 시도당 (실제−기대)']*100:+.2f}%p",
    f"- 같은 개수로 고른 약점이 하반기에도 기대 이하인 비율: 우리 {fmt_pct(wc.loc[OURS,'하반기에도 기대 이하 비율'])}"
    f" vs 단순 정답률 {fmt_pct(wc.loc['단순 정답률 하위','하반기에도 기대 이하 비율'])} vs 무작위 {fmt_pct(wc.loc['무작위 선택','하반기에도 기대 이하 비율'])}",
    f"- 하반기 시도 예측 log-loss 개선(기대 확률만 대비): 보정 점수 {pc.loc['③ ① + 보정 점수 (우리)','①대비 log-loss 개선(%)']:.3f}% vs 단순 정답률 {pc.loc['② ① + 단순 정답률','①대비 log-loss 개선(%)']:.3f}%, "
    f"AUC {pc.loc['① 기대 확률만','AUC']:.3f} → 보정 점수 {pc.loc['③ ① + 보정 점수 (우리)','AUC']:.3f} / 단순 정답률 {pc.loc['② ① + 단순 정답률','AUC']:.3f}",
    "", "## 비교 실험",
    f"- 정답값 최종 해결(기본): 평균 {target_cmp.iloc[0]['정답값 평균']:.3f}, 약점 적중률 우리 {fmt_pct(target_cmp.iloc[0]['약점 적중률(우리)'])} vs 단순 {fmt_pct(target_cmp.iloc[0]['약점 적중률(단순 정답률)'])} vs 무작위 {fmt_pct(target_cmp.iloc[0]['약점 적중률(무작위)'])}",
    f"- 정답값 첫 제출 정답: 평균 {target_cmp.iloc[1]['정답값 평균']:.3f}, 약점 적중률 우리 {fmt_pct(target_cmp.iloc[1]['약점 적중률(우리)'])} vs 단순 {fmt_pct(target_cmp.iloc[1]['약점 적중률(단순 정답률)'])} vs 무작위 {fmt_pct(target_cmp.iloc[1]['약점 적중률(무작위)'])}",
    "- 기간 배정 규칙: " + "; ".join(f"{i} 적중률 {fmt_pct(r['약점 적중률(우리)'])}, 진단 이전 시작 비율 {fmt_pct(r['진단 시도 중 진단 기간 이전에 시작한 비율'])}" for i, r in rule_cmp.iterrows()),
    "", "## 학습 유형",
    f"- K-means k={K_FINAL}, 유형: " + ", ".join(f"{n}({c:,}명)" for n, c in beh['유형'].value_counts().items()),
    f"- 군집-레이팅 연관도 Cramér's V: 구간 내 표준화 {cv_band:.3f} vs 전체 표준화 {cv_glob:.3f}",
    f"- 안정성 ARI 평균 {np.mean(aris):.3f}",
]
(OUT / "결과요약.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
# %% [markdown]
# ### 결과표 엑셀 만들기
# 계산한 표를 발표·책자에 바로 쓸 수 있게 정리합니다: 한글 머리글, 비율은 %, 차이는 %p, 시트마다 설명과 쓰는 곳, 맨 앞에 목차.

# %%
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# 결과표 시트: (원래 표 이름, 시트 이름, 설명, 발표·책자 위치)
RESULT_SHEETS = [
    ("시도 테이블 요약", "1 시도 테이블", "기간별 시도(사용자×문제) 수와 기본 지표", "발표 ②, 책자 2장"),
    ("관찰 부족 효과", "2 관찰 부족", "기간 끝 7일 안에 시작한 시도는 해결률이 낮게 잘림 → 제외 근거", "발표 ②, 책자 4장"),
    ("표1 기대확률 모델 비교", "3 기대확률 모델", "Elo 공식 vs 학습 모델 (AUC 높을수록, log-loss·Brier 낮을수록 좋음)", "발표 ③, 책자 3.2"),
    ("기대확률 보정표(검증)", "4 보정표", "기대 확률 구간별 실제 해결률 (두 값이 비슷할수록 기대 확률을 믿을 수 있음)", "발표 ③, 책자 3.2"),
    ("레이팅 구간별 단순 해결률", "5 구간별 단순 해결률", "레이팅 구간별 태그 단순 해결률", "발표 ③, 책자 3.3"),
    ("레이팅 구간별 기대 대비", "6 구간별 기대 대비", "레이팅 구간별 태그 기대 대비 (실제 해결 ÷ 기대 해결)", "발표 ③, 책자 3.3"),
    ("예시 사용자 태그 진단", "7 사용자 사례", "실제 사용자 한 명의 태그별 진단 (규칙으로 자동 선택)", "발표 ③, 책자 3.3"),
    ("사례 사용자 하반기 확인", "7-2 사례 하반기", "실제 사용자 사례의 상반기 판정과 하반기 기대 대비", "발표 ③⑤, 책자 4장"),
    ("표2a 판정별 하반기 결과", "8 판정별 하반기", "상반기 판정(약점·보통·강점)별 하반기 결과", "발표 ⑤, 책자 4장"),
    ("표2b 약점 선택 방법 비교", "9 약점 선택 비교", "사용자마다 같은 개수의 약점을 방법별로 골라 하반기 결과 비교", "발표 ⑤, 책자 4장"),
    ("표2c 순위 상관", "10 순위 상관", "상반기 태그 순서와 하반기 태그 순서의 일치 정도 (사용자 평균)", "책자 4장"),
    ("표2d 시도 예측 비교", "11 하반기 예측", "하반기 시도 결과 예측 성능 (테스트 사용자)", "발표 ⑤, 책자 4장"),
    ("표3 정답값 기준 비교", "12 정답값 비교", "비교 실험: 최종 해결 vs 첫 제출 정답", "발표 ⑤, 책자 4장"),
    ("표4 기간 배정 규칙 비교", "13 기간 배정 비교", "비교 실험: 첫 제출 / 마지막 제출 / 기준일에서 자르기", "발표 ⑤, 책자 4장"),
    ("행동 지표 기술통계", "14 행동 지표 통계", "진단 대상 사용자의 행동 지표 분포", "책자 3.4"),
    ("표5 학습 유형 프로필", "15 학습 유형", "학습 유형별 인원과 행동 지표 평균", "발표 ④, 책자 5장"),
    ("군집 중심(구간 내 z)", "16 군집 중심", "학습 유형별 행동 지표 (레이팅 구간 안 표준점수, 0=평균)", "발표 ④, 책자 5장"),
    ("군집 수 선택", "17 군집 수 선택", "k별 군집 내 제곱합(inertia)과 실루엣 계수", "발표 ④, 책자 3.4"),
    ("난이도 구간별 결과", "18 난이도 구간별", "전체 사용자의 문제 난이도 구간별 시도 수, 단순 해결률, 기대 해결률, 기대 대비", "발표 ③, 책자 3.3"),
    ("레이팅×난이도 결과", "19 레이팅×난이도", "사용자 레이팅 구간 × 문제 난이도 구간별 해결률과 기대 대비", "발표 ③, 책자 3.3"),
    ("학습 유형×난이도 결과", "20 유형×난이도", "학습 유형별 난이도 구간 시도 비중과 기대 대비", "발표 ④, 책자 5장"),
]
RENAME = {"period": "기간", "index": "구간", "band": "레이팅 구간", "count": "사용자 수", "mean": "평균", "std": "표준편차",
          "min": "최소", "25%": "하위 25%", "50%": "중앙값", "75%": "상위 25%", "max": "최대",
          "Brier개선율": "Brier 개선율", "logloss": "log-loss", "시도수": "시도 수", "사용자수": "사용자 수",
          "해결률": "최종 해결률", "첫제출정답률": "첫 제출 정답률", "평균재시도": "평균 재시도", "관찰부족비율": "관찰 부족 비율",
          "모델포함비율": "모델 포함 비율", "평균기대": "평균 기대 확률", "실제해결률": "실제 해결률"}
INDEX_NAME = {"4 보정표": "기대 확률 구간", "14 행동 지표 통계": "지표"}
PCTP_COLS = ("(실제−기대)", "95% 하한", "95% 상한")          # ×100 해서 %p로 표시
PCT_WORDS = ("률", "율", "비율", "시작", "대비", "평균기대", "정답값 평균", "포기율", "꾸준함", "dp", "greedy", "math")


def number_format(sheet, col, s):
    if sheet == "16 군집 중심":
        return "0.00"
    if sheet == "14 행동 지표 통계":
        return "#,##0" if col == "사용자 수" else "#,##0.000"
    if any(k in col for k in PCTP_COLS):
        return '+0.00"%p";-0.00"%p";0.00"%p"'
    if "개선(%)" in col or "개선%" in col:
        return '0.00"%"'
    if col in ("AUC", "log-loss", "Brier", "기대모델 AUC(검증)") or "순위상관" in col:
        return "0.000"
    if col in ("해결", "시도"):
        return "#,##0"
    if col in ("z값",):
        return "0.00"
    if any(k in col for k in ("평균 재시도", "평균재시도", "주당 시도 수", "재제출 간격", "태그 다양성", "기대 해결 수")):
        return "0.00"
    if any(k in col for k in ("도전 성향", "레이팅")):
        return "#,##0"
    if any(k in col for k in PCT_WORDS):
        return "0.0%"
    if pd.api.types.is_integer_dtype(s) or col.endswith("수"):
        return "#,##0"
    return "0.000"


def export_result_tables(tables, path):
    wb = Workbook()
    toc = wb.active
    toc.title = "목차"
    head_fill = PatternFill("solid", fgColor="DCE6F2")
    thin = Side(style="thin", color="B7B7B7")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    toc["A1"] = "Codeforces 풀이 이력 분석 결과표"
    toc["A1"].font = Font(bold=True, size=14)
    toc["A2"] = f"데이터: 제출 {RESULTS['n_submissions']:,}건, 사용자 {RESULTS['n_users']:,}명 / 정답값: {TARGET} / 진단 2024 상반기 → 검증 2024 하반기"
    for j, h in enumerate(["시트", "내용", "쓰는 곳"], 1):
        c = toc.cell(row=4, column=j, value=h)
        c.font, c.fill, c.border = Font(bold=True), head_fill, box
    r = 5
    for key, sheet, desc, use in RESULT_SHEETS:
        if key not in tables:
            continue
        for j, v in enumerate([sheet, desc, use], 1):
            toc.cell(row=r, column=j, value=v).border = box
        r += 1
        df = tables[key].copy()
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns="Unnamed: 0")
        df = df.rename(columns={"index": INDEX_NAME.get(sheet, "구간")}).rename(columns=RENAME)
        if key == "예시 사용자 태그 진단" and "example_user" in RESULTS:
            eu = RESULTS["example_user"]
            desc = f"{desc}: {eu['handle']} (최근 레이팅 {eu['rating']:.0f})"
        for col in df.columns:
            if any(k in str(col) for k in PCTP_COLS) and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] * 100
        ws = wb.create_sheet(sheet)
        ws["A1"] = sheet[sheet.find(" ") + 1:]
        ws["A1"].font = Font(bold=True, size=13)
        ws["A2"] = desc
        ws["A2"].font = Font(color="595959")
        for j, col in enumerate(df.columns, 1):
            c = ws.cell(row=4, column=j, value=str(col))
            c.font, c.fill, c.border = Font(bold=True), head_fill, box
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            fmt = number_format(sheet, str(col), df[col])
            for i, v in enumerate(df[col].tolist(), 5):
                v = None if (isinstance(v, float) and np.isnan(v)) else v
                cell = ws.cell(row=i, column=j, value=v.item() if hasattr(v, "item") else v)
                cell.border = box
                if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                    cell.number_format = fmt
            width = max([len(str(col))] + [len(f"{x:,.3f}" if isinstance(x, float) else str(x)) for x in df[col].tolist()[:50]])
            ws.column_dimensions[get_column_letter(j)].width = min(max(10, width * 1.3 + 2), 34)
        ws.row_dimensions[4].height = 34
        ws.freeze_panes = "B5"
    for col, w in zip("ABC", (22, 70, 20)):
        toc.column_dimensions[col].width = w
    wb.save(path)


export_result_tables(TABLES, OUT / "결과표.xlsx")
(OUT / "결과값_전체.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("\n저장 완료:", sorted(p.name for p in OUT.iterdir()))

# %% [markdown]
# ## 10. 결과 대시보드 (HTML 파일)
# 위 결과를 한 화면에서 볼 수 있는 대시보드를 `분석결과/결과_대시보드.html`로 저장합니다.
# 인터넷 없이도 브라우저(크롬·엣지·사파리)에서 더블클릭으로 열립니다. 발표 화면에 띄워도 됩니다.
# 숫자는 모두 이 노트북이 계산한 값(TABLES, RESULTS)에서 가져옵니다.

# %%
def r6(x):
    return None if pd.isna(x) else round(float(x), 6)


def dash_datasets():
    """대시보드에 넣을 표들 (모두 이 노트북의 계산 결과)"""
    T = TABLES
    bj = VAL["by_judge"][VAL["by_judge"]["방법"] == OURS]
    att_sum = T["시도 테이블 요약"]
    kpi = [("submissions", RESULTS["n_submissions"]), ("users", RESULTS["n_users"]), ("merge_rate", RESULTS["merge_rate"]),
           ("submissions_used", RESULTS["n_sub_used"]), ("attempts", int(att_sum["시도수"].sum())),
           ("diag_users", RESULTS["n_diag_users"]), ("diag_attempts", RESULTS["n_diag_attempts"]),
           ("valid_pairs", VAL["n_pairs"]), ("valid_users", VAL["n_users"]),
           ("judge_weak", RESULTS["judge_dist"]["기대 대비 z"].get("약점", 0)), ("judge_strong", RESULTS["judge_dist"]["기대 대비 z"].get("강점", 0)),
           ("judge_normal", RESULTS["judge_dist"]["기대 대비 z"].get("보통", 0)), ("judge_unseen", RESULTS["judge_dist"]["기대 대비 z"].get("미경험", 0)),
           ("cluster_k", K_FINAL), ("cramers_v_band", cv_band), ("cramers_v_global", cv_glob), ("cluster_ari", float(np.mean(aris)))]
    so = T["관찰 부족 효과"].set_index(T["관찰 부족 효과"].columns[0])
    p = att_sum.rename(columns={att_sum.columns[0]: "period", "시도수": "attempts", "사용자수": "users", "해결률": "solve_rate",
                                "첫제출정답률": "first_correct_rate", "평균재시도": "mean_retries"})
    p["solve_before_last7"] = p["period"].map(so["7일 이전 시작"])
    p["solve_last7"] = p["period"].map(so["마지막 7일 안 시작"])
    p["period"] = p["period"].map({"모델학습": "모델 학습 (2023 하반기)", "진단": "진단 (2024 상반기)", "검증": "검증 (2024 하반기)"})
    md = T["표1 기대확률 모델 비교"].rename(columns={"기간": "period", "모델": "model", "Brier개선율": "brier_skill"})
    md["key"] = md["period"] + " · " + md["model"]
    cal = T["기대확률 보정표(검증)"].rename(columns={"index": "bin", "시도수": "attempts", "평균기대": "expected", "실제해결률": "actual"})
    order = CONFIG["RATING_BAND_NAMES"]
    a = T["레이팅 구간별 단순 해결률"].melt(id_vars="band", var_name="tag", value_name="raw_rate")
    b = T["레이팅 구간별 기대 대비"].melt(id_vars="band", var_name="tag", value_name="vs_expected")
    bt = a.merge(b, on=["band", "tag"]).assign(band=lambda d: d["band"].astype(str))
    bt["band_order"] = bt["band"].map(order.index)
    uc = T["예시 사용자 태그 진단"].rename(columns={"태그": "tag", "시도": "attempts", "해결": "solved", "단순 해결률": "raw_rate",
                                                  "기대 해결 수": "expected", "기대 대비": "vs_expected", "z값": "z", "판정": "judge"})
    bjr = bj.rename(columns={"상반기 판정": "judge", "쌍 수": "pairs", "하반기 시도당 (실제−기대)": "resid", "95% 하한": "lo", "95% 상한": "hi"})
    wc = T["표2b 약점 선택 방법 비교"].rename(columns={"약점 고르는 방법": "method", "고른 쌍 수": "pairs", "하반기 시도당 (실제−기대)": "resid",
                                                   "95% 하한": "lo", "95% 상한": "hi", "하반기에도 기대 이하 비율": "hit_rate"})
    pc = T["표2d 시도 예측 비교"].rename(columns={"모델": "model", "①대비 log-loss 개선(%)": "logloss_gain_pct"})
    cols = {"정답값": "target", "정답값 평균": "target_mean", "기대모델 AUC(검증)": "model_auc", "약점→하반기 시도당(실제−기대)": "weak_resid",
            "강점→하반기 시도당(실제−기대)": "strong_resid", "약점 적중률(우리)": "hit_ours", "약점 적중률(단순 정답률)": "hit_raw",
            "약점 적중률(무작위)": "hit_random", "예측 개선%(우리)": "gain_ours", "예측 개선%(단순 정답률)": "gain_raw"}
    tc = T["표3 정답값 기준 비교"].rename(columns=cols)
    tc["target"] = ["최종 해결 (기본)" if "solved" in str(x) else "첫 제출 정답" for x in tc["target"]]
    rc = T["표4 기간 배정 규칙 비교"].rename(columns={**cols, "규칙": "rule", "진단 기간 시도 수": "diag_attempts",
                                                  "진단 시도 중 진단 기간 이전에 시작한 비율": "started_before", "검증 시도 중 기준일 이전에 이미 시도한 비율": "seen_before"})
    prof = TABLES["표5 학습 유형 프로필"]
    cz = TABLES["군집 중심(구간 내 z)"]
    feats = []
    for _, row in cz.iterrows():
        pr = prof[prof["유형"] == row["유형"]].iloc[0]
        for i, f in enumerate(CLUSTER_FEATURES):
            feats.append({"type": row["유형"], "feature": f, "feature_order": i, "z": r6(row[f]), "mean": r6(pr[f]), "users": int(pr["사용자 수"])})
    cl = prof.rename(columns={"유형": "type", "사용자 수": "users", "전체 기대 대비": "vs_expected", "최근 레이팅": "rating",
                              "하반기 레이팅 변화": "rating_change_h2", "첫 제출 정답률": "first_correct", "평균 재시도": "retries"}).sort_values("users", ascending=False)

    def rows(df, keep):
        out = []
        for rec in df[keep].to_dict("records"):
            out.append({k: (r6(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v)) for k, v in rec.items()})
        return out
    return {
        "kpi": [{"metric": k, "value": r6(v)} for k, v in kpi],
        "periods": rows(p, ["period", "attempts", "users", "solve_rate", "first_correct_rate", "mean_retries", "solve_before_last7", "solve_last7"]),
        "models": rows(md, ["key", "AUC", "logloss", "Brier", "brier_skill"]),
        "calibration": rows(cal, ["bin", "attempts", "expected", "actual"]),
        "band_tag": rows(bt.sort_values(["tag", "band_order"]), ["band", "tag", "raw_rate", "vs_expected", "band_order"]),
        "user_case": rows(uc, ["tag", "attempts", "solved", "raw_rate", "expected", "vs_expected", "z", "judge"]),
        "by_judge": rows(bjr, ["judge", "pairs", "resid", "lo", "hi"]),
        "weak_cmp": rows(wc, ["method", "pairs", "resid", "lo", "hi", "hit_rate"]),
        "pred_cmp": rows(pc, ["model", "AUC", "logloss", "Brier", "logloss_gain_pct"]),
        "target_cmp": rows(tc, list(cols.values())),
        "rule_cmp": rows(rc, ["rule", "diag_attempts", "started_before", "seen_before", "weak_resid", "hit_ours", "hit_raw", "hit_random"]),
        "clusters": rows(cl, ["type", "users", "rating", "vs_expected", "first_correct", "retries", "rating_change_h2"]),
        "cluster_features": feats,
        "k_select": rows(k_sel, ["k", "inertia", "silhouette"]),
        "diff_band": rows(diff_band.rename(columns={"난이도 구간": "band", "시도수": "attempts", "사용자수": "users", "시도 비중": "share",
                                                    "단순 해결률": "raw_rate", "기대 해결률": "expected_rate", "기대 대비": "vs_expected",
                                                    "첫제출정답률": "first_correct_rate"}).astype({"band": str}),
                          ["band", "attempts", "users", "share", "raw_rate", "expected_rate", "vs_expected", "first_correct_rate"]),
        "diff_rating": rows(rd.assign(rating_order=rd["레이팅 구간"].cat.codes, diff_order=rd["난이도 구간"].cat.codes)
                            .rename(columns={"레이팅 구간": "rating_band", "난이도 구간": "diff_band", "시도수": "attempts",
                                             "단순 해결률": "raw_rate", "기대 대비": "vs_expected"}).astype({"rating_band": str, "diff_band": str}),
                            ["rating_band", "diff_band", "rating_order", "diff_order", "attempts", "raw_rate", "vs_expected"]),
        "diff_type": rows(type_diff.assign(diff_order=type_diff["난이도 구간"].cat.codes)
                          .rename(columns={"유형": "type", "난이도 구간": "diff_band", "시도수": "attempts", "시도 비중": "share", "기대 대비": "vs_expected"})
                          .astype({"diff_band": str}), ["type", "diff_band", "diff_order", "attempts", "share", "vs_expected"]),
        "case_h2": rows(TABLES["사례 사용자 하반기 확인"].rename(columns={"태그": "tag", "상반기 시도": "n", "상반기 기대 대비": "ratio", "상반기 판정": "judge",
                                                                    "하반기 시도": "n_v", "하반기 해결": "solved_v", "하반기 기대 대비": "ratio_v"}),
                        ["tag", "judge", "n", "ratio", "n_v", "solved_v", "ratio_v"]),
    }


DASH_PAGE = r'''<style>
.cf { font-family: var(--font-anthropic-sans); display: flex; flex-direction: column; gap: var(--cds-gap-lg); }
.cf-head { display: flex; flex-direction: column; gap: var(--cds-gap-sm); }
.cf-title { font-family: var(--font-anthropic-serif); font-weight: 400; font-size: var(--cds-font-size-title); margin: 0; text-wrap: balance; }
.cf-take { color: var(--color-fg-muted); font-size: var(--cds-font-size-body); margin: 0; max-width: 70ch; }
.cf-take b, .cf b { font-weight: 600; color: var(--color-fg); }
.cf-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 11rem), 1fr)); gap: var(--cds-gap-md); }
.cf-kpi { background: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: var(--cds-pad-md); display: flex; flex-direction: column; gap: var(--cds-gap-xs); min-width: 0; }
.cf-kpi .lab { font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); }
.cf-kpi .num { font-family: var(--font-anthropic-serif); font-weight: 400; font-size: var(--cds-font-size-title); font-variant-numeric: tabular-nums; }
.cf-kpi .sub { font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); font-variant-numeric: tabular-nums; }
.cf-tabs { display: flex; flex-wrap: wrap; gap: var(--cds-gap-xs); border-bottom: 1px solid var(--color-border-line); }
.cf-tab { appearance: none; background: none; border: 0; border-bottom: 2px solid transparent; padding: var(--cds-pad-sm) var(--cds-pad-md); font: inherit; font-size: var(--cds-font-size-body); color: var(--color-fg-muted); cursor: pointer; margin-bottom: -1px; }
.cf-tab[aria-selected="true"] { color: var(--color-fg); border-bottom-color: var(--color-fg); font-weight: 500; }
.cf-tab:focus-visible { outline: 2px solid var(--color-fg); outline-offset: 2px; }
.cf-sec { display: flex; flex-direction: column; gap: var(--cds-gap-md); }
.cf-intro { margin: 0; color: var(--color-fg-muted); max-width: 75ch; font-size: var(--cds-font-size-body); }
.cf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr)); gap: var(--cds-gap-md); }
.cf-card { background: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: var(--cds-pad-lg); display: flex; flex-direction: column; gap: var(--cds-gap-sm); min-width: 0; }
.cf-card h3 { margin: 0; font-size: var(--cds-font-size-heading); font-weight: var(--cds-font-weight-medium); }
.cf-card .note { margin: 0; font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); }
.cf-chart { position: relative; min-height: 220px; }
.cf-tw { overflow-x: auto; }
.cf-table { width: 100%; border-collapse: collapse; font-size: var(--cds-font-size-caption); font-variant-numeric: tabular-nums; }
.cf-table th, .cf-table td { padding: 6px 8px; border-bottom: 1px solid var(--color-border-line); text-align: right; white-space: nowrap; }
.cf-table th:first-child, .cf-table td:first-child { text-align: left; white-space: normal; min-width: 10rem; }
.cf-table th { color: var(--color-fg-muted); font-weight: 500; }
.cf-table tr.hl td { font-weight: 600; }
.cf-tip { position: absolute; pointer-events: none; background: var(--cds-surface-popover); border: 1px solid var(--cds-border); border-radius: var(--cds-radius); box-shadow: var(--cds-shadow-popover); padding: var(--cds-pad-xs) var(--cds-pad-sm); font-size: var(--cds-font-size-caption); white-space: nowrap; z-index: 2; }
.cf-tip .t { color: var(--cds-text-secondary); }
.cf-legend { display: flex; flex-wrap: wrap; gap: var(--cds-gap-sm); font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); }
.cf-legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; background: var(--sw); vertical-align: -1px; }
.cf-chart text { fill: var(--color-fg-muted); font-size: 11px; font-family: var(--font-anthropic-sans); }
.cf-chart .lbl { fill: var(--color-fg); }
.cf-fail { display: flex; align-items: center; justify-content: center; gap: 6px; min-height: inherit; color: var(--color-fg-muted); font-size: var(--cds-font-size-caption); }
.cf-fail::before { content: ""; width: .375rem; height: .375rem; border-radius: 50%; background: var(--color-bad); }
[data-dash-source-state=empty] { color: transparent; }
[data-dash-source-state=refreshing] { opacity: .5; transition: opacity var(--cds-dur-slow); }
</style>

<div class="cf">
  <header class="cf-head">
    <h1 class="cf-title" id="top">Codeforces 풀이 진단 결과</h1>
    <p class="cf-take" id="takeaway">
      상반기에 <b>약점</b>으로 진단한 태그는 하반기에도 시도당
      <b><span class="dash-skeleton" data-source="by_judge" data-field="resid" data-where="judge=약점">—</span>p</b>,
      <b>강점</b> 태그는 <b><span class="dash-skeleton" data-source="by_judge" data-field="resid" data-where="judge=강점">—</span>p</b>로 판정 순서가 유지됐다.
      같은 개수로 약점을 골랐을 때 하반기 적중률은 우리 방법
      <b class="dash-skeleton" data-source="weak_cmp" data-field="hit_rate" data-where="method=레이팅 보정 진단 (z값)">—</b>,
      단순 정답률 <b class="dash-skeleton" data-source="weak_cmp" data-field="hit_rate" data-where="method=단순 정답률 하위">—</b>이다.
    </p>
  </header>

  <div class="cf-kpis">
    <div class="cf-kpi"><span class="lab">전체 제출</span><span class="num dash-skeleton" data-source="kpi" data-field="value" data-where="metric=submissions" data-fmt="compact">—</span><span class="sub">사용자 <span class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=users" data-fmt="int">—</span>명</span></div>
    <div class="cf-kpi"><span class="lab">분석 시도 (사용자×문제, 2023H2~2024)</span><span class="num dash-skeleton" data-source="kpi" data-field="value" data-where="metric=attempts" data-fmt="compact">—</span><span class="sub">공식 문제 병합률 <span class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=merge_rate" data-fmt="pct">—</span></span></div>
    <div class="cf-kpi"><span class="lab">진단 대상 사용자 (2024 상반기)</span><span class="num dash-skeleton" data-source="kpi" data-field="value" data-where="metric=diag_users" data-fmt="int">—</span><span class="sub">시도 <span class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=diag_attempts" data-fmt="int">—</span>개</span></div>
    <div class="cf-kpi"><span class="lab">검증 쌍 (사용자×태그, 2024 하반기)</span><span class="num dash-skeleton" data-source="kpi" data-field="value" data-where="metric=valid_pairs" data-fmt="int">—</span><span class="sub">사용자 <span class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=valid_users" data-fmt="int">—</span>명</span></div>
    <div class="cf-kpi"><span class="lab">학습 유형</span><span class="num"><span class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=cluster_k" data-fmt="int">—</span>개</span><span class="sub">안정성 ARI <span class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=cluster_ari" data-fmt="f3">—</span></span></div>
  </div>

  <nav class="cf-tabs" role="tablist" aria-label="분석 단계">
    <button class="cf-tab" role="tab" data-tab="data" id="tab-data">① 데이터·시도</button>
    <button class="cf-tab" role="tab" data-tab="model" id="tab-model">② 기대 확률</button>
    <button class="cf-tab" role="tab" data-tab="diag" id="tab-diag">③ 태그·난이도 진단</button>
    <button class="cf-tab" role="tab" data-tab="valid" id="tab-valid">④ 검증</button>
    <button class="cf-tab" role="tab" data-tab="exp" id="tab-exp">⑤ 비교 실험</button>
    <button class="cf-tab" role="tab" data-tab="type" id="tab-type">⑥ 학습 유형</button>
  </nav>

  <section class="cf-sec" data-panel="data" id="data">
    <p class="cf-intro">제출 한 건이 아니라 <b>한 사람이 한 문제를 시도한 것</b>을 한 단위로 본다. 해결 여부는 같은 기간 안 OK 여부, 레이팅과 기간은 첫 제출 시점 기준이다. 기간 끝 7일 안에 시작한 시도는 관찰이 부족해 제외한다.</p>
    <div class="cf-card" id="card-periods">
      <h3>기간별 시도 테이블</h3>
      <p class="note">모델 학습 2023 하반기 → 진단 2024 상반기 → 검증 2024 하반기</p>
      <div class="cf-tw"><table class="cf-table" id="tbl-periods" data-source="periods" data-row-key="period"><thead><tr>
        <th data-field="period">기간</th><th data-field="attempts">시도 수</th><th data-field="users">사용자 수</th><th data-field="solve_rate">최종 해결률</th><th data-field="first_correct_rate">첫 제출 정답률</th><th data-field="mean_retries">평균 재시도</th><th data-field="solve_before_last7">해결률 (7일 이전 시작)</th><th data-field="solve_last7">해결률 (마지막 7일 시작)</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </section>

  <section class="cf-sec" data-panel="model" id="model" hidden>
    <p class="cf-intro">"이 레이팅인 사람이 이 난이도 문제를 풀 확률"을 계산한다. 2023 하반기로 학습하고, 2024 상반기로 모델을 고르고, 2024 하반기에서 확인한다.</p>
    <div class="cf-grid">
      <div class="cf-card" id="card-models">
        <h3>Elo 공식 vs 학습 모델</h3>
        <p class="note">AUC는 높을수록, log-loss·Brier는 낮을수록 좋다. Brier 개선율은 평균만 쓸 때 대비.</p>
        <div class="cf-tw"><table class="cf-table" id="tbl-models" data-source="models" data-row-key="key"><thead><tr>
          <th data-field="key">기간 · 모델</th><th data-field="AUC">AUC</th><th data-field="logloss">log-loss</th><th data-field="Brier">Brier</th><th data-field="brier_skill">Brier 개선율</th>
        </tr></thead><tbody></tbody></table></div>
      </div>
      <div class="cf-card" id="card-calib">
        <h3>보정 곡선 (검증 기간, 학습 모델)</h3>
        <p class="note">가로축은 기대 확률, 세로축은 실제 해결률. 점선 위에 붙을수록 기대 확률을 믿을 수 있다.</p>
        <div class="cf-chart dash-skeleton" id="ch-calib" data-source="calibration"></div>
      </div>
    </div>
  </section>

  <section class="cf-sec" data-panel="diag" id="diag" hidden>
    <p class="cf-intro">태그별로 <b>기대 대비 = 실제 해결 수 ÷ 기대 해결 수</b>를 구하고, z값(±1.5)과 최소 시도 5개로 우연을 거른다. 판정 비율: 약점 <b class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=judge_weak" data-fmt="pct">—</b>, 강점 <b class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=judge_strong" data-fmt="pct">—</b>, 보통 <b class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=judge_normal" data-fmt="pct">—</b>, 미경험 <b class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=judge_unseen" data-fmt="pct">—</b>.</p>
    <div class="cf-grid">
      <div class="cf-card" id="card-raw">
        <h3>레이팅 구간별 단순 해결률</h3>
        <p class="note">레이팅이 낮을수록 모든 태그가 낮아 보인다.</p>
        <div class="cf-chart dash-skeleton" id="ch-raw" data-source="band_tag"></div>
      </div>
      <div class="cf-card" id="card-ratio">
        <h3>레이팅 구간별 기대 대비</h3>
        <p class="note">레이팅을 보정하면 100% 근처에서 태그끼리 비교된다.</p>
        <div class="cf-chart dash-skeleton" id="ch-ratio" data-source="band_tag"></div>
      </div>
    </div>
    <div class="cf-card" id="card-case">
      <h3 id="case-title">실제 사용자 사례</h3>
      <p class="note">진단 대상 8,507명 중 규칙(최근 레이팅 900~1250, dp 12문제 이상 '약점' + greedy 15문제 이상 '강점', 약점 수가 가장 적고 레이팅 1000에 가장 가까운 사람)으로 자동 선택한 실제 사용자의 2024 상반기 진단. 막대는 기대 대비, 색은 판정.</p>
      <div class="cf-legend" id="lg-case"></div>
      <div class="cf-chart dash-skeleton" id="ch-case" data-source="user_case"></div>
      <div class="cf-tw"><table class="cf-table" id="tbl-case" data-source="user_case" data-row-key="tag"><thead><tr>
        <th data-field="tag">태그</th><th data-field="attempts">시도</th><th data-field="solved">해결</th><th data-field="raw_rate">단순 해결률</th><th data-field="expected">기대 해결 수</th><th data-field="vs_expected">기대 대비</th><th data-field="z">z값</th><th data-field="judge">판정</th>
      </tr></thead><tbody></tbody></table></div>
      <h3 style="margin-top: var(--cds-gap-md)">같은 사용자의 2024 하반기 확인</h3>
      <p class="note">상반기 판정이 하반기에도 이어졌는지 (하반기 레이팅으로 다시 계산한 기대 대비, 하반기 시도 3개 이상 태그)</p>
      <div class="cf-tw"><table class="cf-table" id="tbl-case-h2" data-source="case_h2" data-row-key="tag"><thead><tr>
        <th data-field="tag">태그</th><th data-field="judge">상반기 판정</th><th data-field="n">상반기 시도</th><th data-field="ratio">상반기 기대 대비</th><th data-field="n_v">하반기 시도</th><th data-field="solved_v">하반기 해결</th><th data-field="ratio_v">하반기 기대 대비</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="cf-grid">
      <div class="cf-card" id="card-diff">
        <h3>난이도 구간별 단순 해결률과 기대 해결률 (전체 사용자)</h3>
        <p class="note">2024 상반기 진단 대상 전체 시도. 막대 위 숫자는 기대 대비와 시도 수.</p>
        <div class="cf-legend" id="lg-diff"></div>
        <div class="cf-chart dash-skeleton" id="ch-diff" data-source="diff_band"></div>
      </div>
      <div class="cf-card" id="card-diffheat">
        <h3>레이팅 구간 × 난이도 구간 기대 대비</h3>
        <p class="note">파랑은 레이팅에 비해 잘 풀고, 빨강은 못 푼다. 시도 300개 미만 칸은 '적음'. 기대 확률 상한(95%) 때문에 쉬운 문제와 높은 레이팅에서는 100%보다 조금 높게 나오므로 칸끼리 비교한다.</p>
        <div class="cf-chart dash-skeleton" id="ch-diffheat" data-source="diff_rating"></div>
      </div>
    </div>
  </section>

  <section class="cf-sec" data-panel="valid" id="valid" hidden>
    <p class="cf-intro">상반기(진단) 판정이 하반기(검증)에도 맞는지 본다. 하반기 기대 확률은 하반기 레이팅으로 다시 계산했다. 막대는 하반기 시도당 (실제 해결 − 기대 해결), 선은 95% 신뢰구간.</p>
    <div class="cf-grid">
      <div class="cf-card" id="card-judge">
        <h3>① 상반기 판정별 하반기 결과</h3>
        <p class="note">약점 &lt; 보통 &lt; 강점 순서가 하반기에도 유지되는지</p>
        <div class="cf-chart dash-skeleton" id="ch-judge" data-source="by_judge"></div>
      </div>
      <div class="cf-card" id="card-weak">
        <h3>② 같은 개수로 약점을 골랐을 때 하반기 적중률</h3>
        <p class="note">하반기에도 기대 이하로 나온 비율. 사용자마다 우리 방법이 고른 개수만큼 다른 방법으로도 골랐다.</p>
        <div class="cf-chart dash-skeleton" id="ch-weak" data-source="weak_cmp"></div>
      </div>
    </div>
    <div class="cf-card" id="card-pred">
      <h3>③ 하반기 시도 예측</h3>
      <p class="note">기대 확률에 상반기 지표를 더해 하반기 해결 여부를 예측 (사용자 절반 학습, 절반 평가)</p>
      <div class="cf-tw"><table class="cf-table" id="tbl-pred" data-source="pred_cmp" data-row-key="model"><thead><tr>
        <th data-field="model">모델</th><th data-field="AUC">AUC</th><th data-field="logloss">log-loss</th><th data-field="Brier">Brier</th><th data-field="logloss_gain_pct">log-loss 개선 (①대비)</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </section>

  <section class="cf-sec" data-panel="exp" id="exp" hidden>
    <p class="cf-intro">같은 방법을 조건만 바꿔 다시 돌렸다. 적중률은 위 ②와 같은 방식(우리 / 단순 정답률 / 무작위)이다.</p>
    <div class="cf-card" id="card-target">
      <h3>정답값: 최종 해결 vs 첫 제출 정답</h3>
      <p class="note">최종 해결은 대부분 1이라(천장 효과) 태그 사이 차이가 작게 나온다.</p>
      <div class="cf-tw"><table class="cf-table" id="tbl-target" data-source="target_cmp" data-row-key="target"><thead><tr>
        <th data-field="target">정답값</th><th data-field="target_mean">정답값 평균</th><th data-field="model_auc">기대 모델 AUC</th><th data-field="weak_resid">약점→하반기</th><th data-field="strong_resid">강점→하반기</th><th data-field="hit_ours">적중률 우리</th><th data-field="hit_raw">단순 정답률</th><th data-field="hit_random">무작위</th><th data-field="gain_ours">예측 개선 우리</th><th data-field="gain_raw">단순 정답률</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
    <div class="cf-card" id="card-rule">
      <h3>기간 배정 규칙: 첫 제출 vs 마지막 제출 vs 기준일에서 자르기</h3>
      <p class="note">마지막 제출 기준은 기준일 이후 정보로 기간을 정하므로 누수다. 결론은 세 규칙에서 같다.</p>
      <div class="cf-tw"><table class="cf-table" id="tbl-rule" data-source="rule_cmp" data-row-key="rule"><thead><tr>
        <th data-field="rule">규칙</th><th data-field="diag_attempts">진단 시도 수</th><th data-field="started_before">진단 이전 시작</th><th data-field="seen_before">검증 중 이미 본 문제</th><th data-field="weak_resid">약점→하반기</th><th data-field="hit_ours">적중률 우리</th><th data-field="hit_raw">단순 정답률</th><th data-field="hit_random">무작위</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </section>

  <section class="cf-sec" data-panel="type" id="type" hidden>
    <p class="cf-intro">행동 지표 12개를 <b>레이팅 구간 안에서 표준화</b>한 뒤 K-means로 묶었다. 레이팅과의 연관도(Cramér's V)는 구간 내 표준화 <b class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=cramers_v_band" data-fmt="f3">—</b>, 전체 표준화 <b class="dash-skeleton" data-source="kpi" data-field="value" data-where="metric=cramers_v_global" data-fmt="f3">—</b>. 유형 이름은 각 군집에서 가장 두드러진 지표 2개로 붙인 자동 이름(결과표 15번)을 근거로 정했다: 꾸준한 다작형, 안전 지향형(쉬운 문제·익숙한 유형 위주), 도전적 시행착오형, 효율 개선 필요형(시간·메모리 초과가 많음).</p>
    <div class="cf-card" id="card-heat">
      <h3>유형별 행동 지표 (레이팅 구간 안 표준점수)</h3>
      <p class="note">0이 평균. 빨강은 평균보다 높고 파랑은 낮다. 칸에 올리면 실제 평균값이 보인다.</p>
      <div class="cf-chart dash-skeleton" id="ch-heat" data-source="cluster_features"></div>
    </div>
    <div class="cf-card" id="card-typediff">
      <h3>학습 유형별 시도한 문제의 난이도 분포</h3>
      <p class="note">2024 상반기 시도 중 난이도 구간별 비중. 막대에 올리면 그 구간의 기대 대비가 보인다.</p>
      <div class="cf-legend" id="lg-typediff"></div>
      <div class="cf-chart dash-skeleton" id="ch-typediff" data-source="diff_type"></div>
    </div>
    <div class="cf-grid">
      <div class="cf-card" id="card-clusters">
        <h3>유형별 요약</h3>
        <p class="note">하반기 레이팅 변화는 상관일 뿐 인과가 아니다.</p>
        <div class="cf-tw"><table class="cf-table" id="tbl-clusters" data-source="clusters" data-row-key="type"><thead><tr>
          <th data-field="type">유형</th><th data-field="users">인원</th><th data-field="rating">평균 레이팅</th><th data-field="vs_expected">전체 기대 대비</th><th data-field="first_correct">첫 제출 정답률</th><th data-field="retries">평균 재시도</th><th data-field="rating_change_h2">하반기 레이팅 변화</th>
        </tr></thead><tbody></tbody></table></div>
      </div>
      <div class="cf-card" id="card-k">
        <h3>군집 수 선택 (실루엣 계수)</h3>
        <p class="note">k=2가 가장 높지만 유형 구분을 위해 4~6 중 가장 높은 k=4를 골랐다.</p>
        <div class="cf-chart dash-skeleton" id="ch-k" data-source="k_select"></div>
      </div>
    </div>
  </section>
</div>

<script>
(function () {
  const F = {
    int: d3.format(',.0f'), pct: d3.format('.1%'), f3: d3.format('.3f'), f2: d3.format('.2f'),
    compact: (v) => new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(v),
    pp: (v) => (v >= 0 ? '+' : '−') + d3.format('.2f')(Math.abs(v * 100)) + '%p',
    gain: (v) => d3.format('.2f')(v) + '%',
    ppMark: (v) => (v >= 0 ? '+' : '-') + d3.format('.2f')(Math.abs(v * 100)) + '%',
  };
  const num = (x) => (x === '' || x == null ? NaN : Number(x));
  const JUDGE = { '약점': 'var(--cds-chart-status-critical)', '강점': 'var(--color-ok)', '보통': 'var(--cds-chart-muted)', '미경험': 'var(--cds-chart-muted)' };

  // 탭
  const tabs = [...document.querySelectorAll('.cf-tab')];
  const panels = [...document.querySelectorAll('[data-panel]')];
  function show(name) {
    tabs.forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
    panels.forEach((p) => { p.hidden = p.dataset.panel !== name; });
    if (window.cfDraw) window.cfDraw();
  }
  tabs.forEach((t) => t.addEventListener('click', () => { show(t.dataset.tab); dash.setLink(t.dataset.tab); }));
  const start = location.hash.slice(1);
  show(panels.some((p) => p.dataset.panel === start) ? start : 'data');

  function fail(el) { el.innerHTML = ''; const d = document.createElement('div'); d.className = 'cf-fail'; d.textContent = 'Failed to load'; el.appendChild(d); }
  function tip(box) {
    let t = box.querySelector('.cf-tip');
    if (!t) { t = document.createElement('div'); t.className = 'cf-tip'; t.hidden = true; box.appendChild(t); }
    return {
      show(ev, title, line) {
        t.innerHTML = ''; const a = document.createElement('div'); a.className = 't'; a.textContent = title; t.appendChild(a);
        const b = document.createElement('div'); b.textContent = line; t.appendChild(b); t.hidden = false;
        const r = box.getBoundingClientRect(); let x = ev.clientX - r.left + 12; const y = ev.clientY - r.top + 12;
        if (x + t.offsetWidth > box.clientWidth) x = ev.clientX - r.left - t.offsetWidth - 12;
        t.style.left = Math.max(0, x) + 'px'; t.style.top = y + 'px';
      },
      hide() { t.hidden = true; },
    };
  }
  function svgFor(el, h) {
    el.querySelectorAll('svg').forEach((s) => s.remove());
    const w = Math.max(280, el.clientWidth);
    return { svg: d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${w} ${h}`).attr('height', h), w };
  }
  function measure(svg, texts) {
    const g = svg.append('g'); let m = 0;
    texts.forEach((s) => { const t = g.append('text').text(s); m = Math.max(m, t.node().getComputedTextLength()); });
    g.remove(); return m;
  }
  function gridAxis(g, axis) { g.call(axis).attr('font-size', null).attr('font-family', null); g.select('.domain').attr('stroke', 'var(--cds-chart-axis)'); g.selectAll('.tick line').attr('stroke', 'var(--cds-chart-grid)'); }

  function fillMarks() {
    document.querySelectorAll('[data-source][data-field][data-where]').forEach((el) => {
      const r = dash.data(el.dataset.source);
      if (r.status !== 'ok') return;
      const [k, v] = el.dataset.where.split('=');
      const row = r.data.find((d) => String(d[k]) === v);
      if (!row) { el.textContent = '—'; return; }
      const x = num(row[el.dataset.field]);
      const fmt = el.dataset.fmt || (el.dataset.source === 'by_judge' ? 'ppMark' : 'pct');
      el.textContent = F[fmt](x);
      el.classList.remove('dash-skeleton');
    });
  }

  function table(id, src, cols, opts = {}) {
    const el = document.getElementById(id), r = dash.data(src), tb = el.querySelector('tbody');
    if (r.status !== 'ok') { tb.innerHTML = ''; return; }
    tb.innerHTML = '';
    r.data.forEach((row) => {
      const tr = document.createElement('tr');
      if (opts.hl && opts.hl(row)) tr.className = 'hl';
      cols.forEach((c, i) => {
        const td = document.createElement('td');
        const v = row[c[0]];
        td.textContent = i === 0 || !c[1] ? v : (Number.isFinite(num(v)) ? F[c[1]](num(v)) : '—');
        if (c[2]) td.style.color = c[2](row);
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  }

  function drawCalib() {
    const el = document.getElementById('ch-calib'), r = dash.data('calibration');
    if (r.status === 'error') return fail(el); if (r.status !== 'ok') return;
    el.classList.remove('dash-skeleton');
    const rows = r.data.map((d) => ({ bin: d.bin, e: num(d.expected), a: num(d.actual), n: num(d.attempts) }));
    const h = 240, { svg, w } = svgFor(el, h), m = { t: 10, r: 14, b: 32, l: 40 };
    const lo = Math.floor(d3.min(rows, (d) => Math.min(d.e, d.a)) * 10) / 10;
    const x = d3.scaleLinear([lo, 1], [m.l, w - m.r]), y = d3.scaleLinear([lo, 1], [h - m.b, m.t]);
    gridAxis(svg.append('g').attr('transform', `translate(0,${h - m.b})`), d3.axisBottom(x).ticks(5).tickFormat(d3.format('.0%')).tickSize(-(h - m.t - m.b)));
    gridAxis(svg.append('g').attr('transform', `translate(${m.l},0)`), d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')).tickSize(-(w - m.l - m.r)));
    svg.append('line').attr('x1', x(lo)).attr('y1', y(lo)).attr('x2', x(1)).attr('y2', y(1)).attr('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
    svg.append('path').datum(rows).attr('fill', 'none').attr('stroke', dash.colors[0]).attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('d', d3.line((d) => x(d.e), (d) => y(d.a)));
    const tp = tip(el);
    svg.selectAll('circle').data(rows).join('circle').attr('cx', (d) => x(d.e)).attr('cy', (d) => y(d.a)).attr('r', 5).attr('fill', dash.colors[0])
      .on('mousemove', (ev, d) => tp.show(ev, `기대 확률 ${d.bin} · ${F.int(d.n)}개`, `기대 ${F.pct(d.e)} → 실제 ${F.pct(d.a)}`)).on('mouseleave', () => tp.hide());
  }

  function drawBand(id, field, fmtTick, ref) {
    const el = document.getElementById(id), r = dash.data('band_tag');
    if (r.status === 'error') return fail(el); if (r.status !== 'ok') return;
    el.classList.remove('dash-skeleton');
    const rows = r.data.map((d) => ({ band: d.band, o: num(d.band_order), tag: d.tag, v: num(d[field]) })).sort((a, b) => a.o - b.o);
    const bands = [...new Set(rows.map((d) => d.band))], tags = [...new Set(rows.map((d) => d.tag))];
    const h = 240, { svg, w } = svgFor(el, h);
    const endW = measure(svg, tags.map((t) => t + ' 100.0%')) + 10;
    const m = { t: 10, r: endW, b: 30, l: 44 };
    const ext = d3.extent(rows, (d) => d.v); const pad = (ext[1] - ext[0]) * 0.15;
    const x = d3.scalePoint(bands, [m.l, w - m.r]).padding(0.2), y = d3.scaleLinear([Math.min(ext[0] - pad, ref ?? Infinity), Math.max(ext[1] + pad, ref ?? -Infinity)], [h - m.b, m.t]).nice();
    gridAxis(svg.append('g').attr('transform', `translate(0,${h - m.b})`), d3.axisBottom(x).tickSize(0).tickPadding(8));
    gridAxis(svg.append('g').attr('transform', `translate(${m.l},0)`), d3.axisLeft(y).ticks(5).tickFormat(fmtTick).tickSize(-(w - m.l - m.r)));
    if (ref != null) svg.append('line').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(ref)).attr('y2', y(ref)).attr('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
    const tp = tip(el), ends = [];
    tags.forEach((t, i) => {
      const s = rows.filter((d) => d.tag === t);
      svg.append('path').datum(s).attr('fill', 'none').attr('stroke', dash.colors[i]).attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('d', d3.line((d) => x(d.band), (d) => y(d.v)));
      svg.append('g').selectAll('circle').data(s).join('circle').attr('cx', (d) => x(d.band)).attr('cy', (d) => y(d.v)).attr('r', 4).attr('fill', dash.colors[i])
        .on('mousemove', (ev, d) => tp.show(ev, `레이팅 ${d.band}`, `${t} ${F.pct(d.v)}`)).on('mouseleave', () => tp.hide());
      const last = s[s.length - 1]; ends.push({ t, i, y: y(last.v), v: last.v });
    });
    ends.sort((a, b) => a.y - b.y); for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 13) ends[k].y = ends[k - 1].y + 13;
    ends.forEach((e) => svg.append('text').attr('class', 'lbl').attr('x', w - m.r + 8).attr('y', e.y).attr('dy', 4).style('fill', dash.colors[e.i]).text(`${e.t} ${F.pct(e.v)}`));
  }

  function hbars(id, src, opt) {
    const el = document.getElementById(id), r = dash.data(src);
    if (r.status === 'error') return fail(el); if (r.status !== 'ok') return;
    el.classList.remove('dash-skeleton');
    const rows = r.data.map(opt.map);
    const bh = 22, gap = 8, m = { t: 6, b: 26 };
    const h = m.t + m.b + rows.length * (bh + gap);
    el.style.minHeight = h + 'px';
    const { svg, w } = svgFor(el, h);
    const lw = measure(svg, rows.map((d) => d.label)) + 10, vw = measure(svg, rows.map((d) => opt.fmt(d.v))) + 14;
    const ml = lw + (rows.some((d) => d.v < 0) ? vw : 0), mr = vw;
    const dom = opt.domain(rows);
    const x = d3.scaleLinear(dom, [ml, w - mr]);
    const y = (i) => m.t + i * (bh + gap);
    gridAxis(svg.append('g').attr('transform', `translate(0,${h - m.b})`), d3.axisBottom(x).ticks(Math.max(2, Math.floor((w - ml - mr) / 80))).tickFormat(opt.tick).tickSize(-(h - m.t - m.b)));
    if (opt.ref != null) svg.append('line').attr('x1', x(opt.ref)).attr('x2', x(opt.ref)).attr('y1', m.t).attr('y2', h - m.b).attr('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
    const tp = tip(el), base = x(opt.base ?? dom[0]);
    rows.forEach((d, i) => {
      svg.append('text').attr('class', 'lbl').attr('x', lw - 8).attr('y', y(i) + bh / 2).attr('dy', 4).attr('text-anchor', 'end').text(d.label);
      const x0 = Math.min(base, x(d.v)), x1 = Math.max(base, x(d.v));
      svg.append('rect').attr('x', x0).attr('y', y(i)).attr('width', Math.max(1, x1 - x0)).attr('height', bh).attr('rx', 4).attr('fill', d.color)
        .on('mousemove', (ev) => tp.show(ev, d.label, d.tipText)).on('mouseleave', () => tp.hide());
      if (d.lo != null) {
        svg.append('line').attr('x1', x(d.lo)).attr('x2', x(d.hi)).attr('y1', y(i) + bh / 2).attr('y2', y(i) + bh / 2).attr('stroke', 'var(--color-fg)').attr('stroke-width', 1.5);
      }
      const tx = d.v >= (opt.base ?? dom[0]) ? Math.max(x(d.v), d.hi != null ? x(d.hi) : 0) + 6 : Math.min(x(d.v), d.lo != null ? x(d.lo) : Infinity) - 6;
      svg.append('text').attr('class', 'lbl').attr('x', tx).attr('y', y(i) + bh / 2).attr('dy', 4).attr('text-anchor', d.v >= (opt.base ?? dom[0]) ? 'start' : 'end').text(opt.fmt(d.v));
    });
  }

  function drawK() {
    const el = document.getElementById('ch-k'), r = dash.data('k_select');
    if (r.status === 'error') return fail(el); if (r.status !== 'ok') return;
    el.classList.remove('dash-skeleton');
    const rows = r.data.map((d) => ({ k: num(d.k), s: num(d.silhouette) }));
    const h = 220, { svg, w } = svgFor(el, h), m = { t: 14, r: 16, b: 30, l: 44 };
    const x = d3.scaleLinear(d3.extent(rows, (d) => d.k), [m.l, w - m.r]), y = d3.scaleLinear([0, d3.max(rows, (d) => d.s) * 1.15], [h - m.b, m.t]);
    gridAxis(svg.append('g').attr('transform', `translate(0,${h - m.b})`), d3.axisBottom(x).ticks(rows.length).tickFormat((d) => 'k=' + d).tickSize(0).tickPadding(8));
    gridAxis(svg.append('g').attr('transform', `translate(${m.l},0)`), d3.axisLeft(y).ticks(4).tickFormat(d3.format('.2f')).tickSize(-(w - m.l - m.r)));
    svg.append('path').datum(rows).attr('fill', 'none').attr('stroke', 'var(--cds-chart-muted)').attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('d', d3.line((d) => x(d.k), (d) => y(d.s)));
    const tp = tip(el);
    svg.selectAll('circle').data(rows).join('circle').attr('cx', (d) => x(d.k)).attr('cy', (d) => y(d.s)).attr('r', (d) => (d.k === 4 ? 6 : 4)).attr('fill', (d) => (d.k === 4 ? dash.colors[0] : 'var(--cds-chart-muted)'))
      .on('mousemove', (ev, d) => tp.show(ev, `k=${d.k}`, `실루엣 ${F.f3(d.s)}`)).on('mouseleave', () => tp.hide());
    const sel = rows.find((d) => d.k === 4);
    if (sel) svg.append('text').attr('class', 'lbl').attr('x', x(4)).attr('y', y(sel.s) - 16).attr('text-anchor', 'start').text(`선택 k=4 (${F.f3(sel.s)})`);
  }

  function drawHeat() {
    const el = document.getElementById('ch-heat'), r = dash.data('cluster_features');
    if (r.status === 'error') return fail(el); if (r.status !== 'ok') return;
    el.classList.remove('dash-skeleton');
    const rows = r.data.map((d) => ({ type: d.type, f: d.feature, o: num(d.feature_order), z: num(d.z), mean: num(d.mean), n: num(d.users) }));
    const types = [...new Set(rows.map((d) => d.type))].sort((a, b) => rows.find((d) => d.type === b).n - rows.find((d) => d.type === a).n);
    const feats = [...new Set(rows.sort((a, b) => a.o - b.o).map((d) => d.f))];
    const resolve = (c) => { const p = document.createElement('span'); p.style.color = c; el.appendChild(p); const v = getComputedStyle(p).color; p.remove(); return v; };
    const hot = resolve('var(--cds-chart-status-critical)'), cool = resolve(dash.colors[0]), mid = resolve('var(--color-panel)');
    const { svg: probe } = svgFor(el, 10);
    const lw = measure(probe, types.map((t) => `${t} (${F.int(rows.find((d) => d.type === t).n)}명)`)) + 12;
    const fh = measure(probe, feats) + 10; probe.remove();
    const cellH = 34, h = fh * 0.9 + 8 + types.length * cellH + 12;
    const { svg, w } = svgFor(el, h);
    const x = d3.scaleBand(feats, [lw, w - fh * 0.5 - 4]).padding(0.06), y0 = fh * 0.9 + 8;
    const col = (z) => (z >= 0 ? d3.interpolateRgb(mid, hot)(Math.min(1, z / 1.5)) : d3.interpolateRgb(mid, cool)(Math.min(1, -z / 1.5)));
    feats.forEach((f) => svg.append('text').attr('transform', `translate(${x(f) + x.bandwidth() / 2},${y0 - 6}) rotate(-60)`).attr('text-anchor', 'start').text(f));
    const tp = tip(el);
    const pctF = new Set(['포기율', '첫 제출 정답률', 'WA 비율', 'TLE·MLE 비율', 'RE 비율', 'CE 비율', '꾸준함']);
    types.forEach((t, i) => {
      const n = rows.find((d) => d.type === t).n;
      svg.append('text').attr('class', 'lbl').attr('x', lw - 8).attr('y', y0 + i * cellH + cellH / 2).attr('dy', 4).attr('text-anchor', 'end').text(`${t} (${F.int(n)}명)`);
      rows.filter((d) => d.type === t).forEach((d) => {
        svg.append('rect').attr('x', x(d.f)).attr('y', y0 + i * cellH + 2).attr('width', x.bandwidth()).attr('height', cellH - 4).attr('rx', 3).attr('fill', col(d.z))
          .on('mousemove', (ev) => tp.show(ev, `${t} · ${d.f}`, `평균 ${pctF.has(d.f) ? F.pct(d.mean) : F.f2(d.mean)} (표준점수 ${d3.format('+.2f')(d.z)})`)).on('mouseleave', () => tp.hide());
        if (x.bandwidth() > 30) svg.append('text').attr('class', 'lbl').attr('x', x(d.f) + x.bandwidth() / 2).attr('y', y0 + i * cellH + cellH / 2).attr('dy', 4).attr('text-anchor', 'middle').style('pointer-events', 'none').text(d3.format('+.1f')(d.z));
      });
    });
    el.style.minHeight = h + 'px';
  }

  function drawDiff() {
    const el = document.getElementById('ch-diff'), r = dash.data('diff_band');
    if (r.status === 'error') return fail(el); if (r.status !== 'ok') return;
    el.classList.remove('dash-skeleton');
    const rows = r.data.map((d) => ({ b: d.band, raw: num(d.raw_rate), ex: num(d.expected_rate), ratio: num(d.vs_expected), n: num(d.attempts) }));
    const lg = document.getElementById('lg-diff'); lg.innerHTML = '';
    [['단순 해결률', dash.colors[0]], ['기대 해결률 (레이팅 기준)', 'var(--cds-chart-muted)']].forEach(([t, c]) => { const s = document.createElement('span'); s.style.setProperty('--sw', c); s.textContent = t; lg.appendChild(s); });
    const h = 260, { svg, w } = svgFor(el, h), m = { t: 34, r: 10, b: 28, l: 40 };
    const x = d3.scaleBand(rows.map((d) => d.b), [m.l, w - m.r]).padding(0.25), x2 = d3.scaleBand(['raw', 'ex'], [0, x.bandwidth()]).padding(0.08);
    const lo = Math.max(0, Math.floor(d3.min(rows, (d) => Math.min(d.raw, d.ex)) * 10) / 10 - 0.1);
    const y = d3.scaleLinear([lo, 1], [h - m.b, m.t]);
    gridAxis(svg.append('g').attr('transform', `translate(0,${h - m.b})`), d3.axisBottom(x).tickSize(0).tickPadding(8));
    gridAxis(svg.append('g').attr('transform', `translate(${m.l},0)`), d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')).tickSize(-(w - m.l - m.r)));
    const tp = tip(el);
    rows.forEach((d) => {
      [['raw', d.raw, dash.colors[0]], ['ex', d.ex, 'var(--cds-chart-muted)']].forEach(([k, v, c]) => {
        svg.append('rect').attr('x', x(d.b) + x2(k)).attr('y', y(v)).attr('width', x2.bandwidth()).attr('height', y(lo) - y(v)).attr('rx', 3).attr('fill', c)
          .on('mousemove', (ev) => tp.show(ev, `난이도 ${d.b} · ${F.int(d.n)}개 시도`, `${k === 'raw' ? '단순 해결률' : '기대 해결률'} ${F.pct(v)}`)).on('mouseleave', () => tp.hide());
      });
      svg.append('text').attr('class', 'lbl').attr('x', x(d.b) + x.bandwidth() / 2).attr('y', y(Math.max(d.raw, d.ex)) - 6).attr('text-anchor', 'middle').text(F.pct(d.ratio));
    });
  }

  function drawDiffHeat() {
    const el = document.getElementById('ch-diffheat'), r = dash.data('diff_rating');
    if (r.status === 'error') return fail(el); if (r.status !== 'ok') return;
    el.classList.remove('dash-skeleton');
    const rows = r.data.map((d) => ({ rb: d.rating_band, db: d.diff_band, ro: num(d.rating_order), dO: num(d.diff_order), n: num(d.attempts), raw: num(d.raw_rate), v: num(d.vs_expected) }));
    const rbs = [...new Map(rows.sort((a, b) => a.ro - b.ro).map((d) => [d.rb, 1])).keys()];
    const dbs = [...new Map(rows.slice().sort((a, b) => a.dO - b.dO).map((d) => [d.db, 1])).keys()];
    const resolve = (c) => { const p = document.createElement('span'); p.style.color = c; el.appendChild(p); const v = getComputedStyle(p).color; p.remove(); return v; };
    const hot = resolve('var(--cds-chart-status-critical)'), cool = resolve(dash.colors[0]), mid = resolve('var(--color-panel)'), none = resolve('var(--color-border-line)');
    const col = (v) => (v >= 1 ? d3.interpolateRgb(mid, cool)(Math.min(1, (v - 1) / 0.08)) : d3.interpolateRgb(mid, hot)(Math.min(1, (1 - v) / 0.08)));
    const { svg: probe } = svgFor(el, 10); const lw = measure(probe, rbs) + 14; probe.remove();
    const cellH = 30, top = 22, h = top + rbs.length * cellH + 8, { svg, w } = svgFor(el, h);
    const x = d3.scaleBand(dbs, [lw, w - 4]).padding(0.05);
    dbs.forEach((b) => svg.append('text').attr('x', x(b) + x.bandwidth() / 2).attr('y', 14).attr('text-anchor', 'middle').text(b));
    const tp = tip(el);
    rbs.forEach((rb, i) => {
      svg.append('text').attr('class', 'lbl').attr('x', lw - 8).attr('y', top + i * cellH + cellH / 2).attr('dy', 4).attr('text-anchor', 'end').text(rb);
      dbs.forEach((db) => {
        const d = rows.find((q) => q.rb === rb && q.db === db);
        const enough = d && d.n >= 300;
        svg.append('rect').attr('x', x(db)).attr('y', top + i * cellH + 2).attr('width', x.bandwidth()).attr('height', cellH - 4).attr('rx', 3).attr('fill', enough ? col(d.v) : none)
          .on('mousemove', (ev) => tp.show(ev, `레이팅 ${rb} · 난이도 ${db}`, d ? `${F.int(d.n)}개 시도 · 단순 ${F.pct(d.raw)} · 기대 대비 ${F.pct(d.v)}` : '시도 없음')).on('mouseleave', () => tp.hide());
        if (x.bandwidth() > 34) svg.append('text').attr('class', 'lbl').attr('x', x(db) + x.bandwidth() / 2).attr('y', top + i * cellH + cellH / 2).attr('dy', 4).attr('text-anchor', 'middle').style('pointer-events', 'none').text(enough ? F.pct(d.v) : '적음');
      });
    });
    el.style.minHeight = h + 'px';
  }

  function drawTypeDiff() {
    const el = document.getElementById('ch-typediff'), r = dash.data('diff_type');
    if (r.status === 'error') return fail(el); if (r.status !== 'ok') return;
    el.classList.remove('dash-skeleton');
    const rows = r.data.map((d) => ({ t: d.type, b: d.diff_band, o: num(d.diff_order), s: num(d.share), v: num(d.vs_expected), n: num(d.attempts) }));
    const order = (dash.data('clusters').data || []).map((d) => d.type);
    const types = [...new Set(rows.map((d) => d.t))].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    const bands = [...new Map(rows.slice().sort((a, b) => a.o - b.o).map((d) => [d.b, 1])).keys()];
    const resolve = (c) => { const p = document.createElement('span'); p.style.color = c; el.appendChild(p); const v = getComputedStyle(p).color; p.remove(); return v; };
    const c0 = resolve('var(--color-panel)'), c1 = resolve(dash.colors[0]);
    const colOf = (i) => d3.interpolateRgb(c0, c1)(0.25 + 0.75 * (i / Math.max(1, bands.length - 1)));
    const lg = document.getElementById('lg-typediff'); lg.innerHTML = '';
    bands.forEach((b, i) => { const s = document.createElement('span'); s.style.setProperty('--sw', colOf(i)); s.textContent = '난이도 ' + b; lg.appendChild(s); });
    const { svg: probe } = svgFor(el, 10); const lw = measure(probe, types) + 12; probe.remove();
    const bh = 24, gap = 10, h = types.length * (bh + gap) + 26, { svg, w } = svgFor(el, h);
    const x = d3.scaleLinear([0, 1], [lw, w - 22]);
    gridAxis(svg.append('g').attr('transform', `translate(0,${h - 22})`), d3.axisBottom(x).ticks(5).tickFormat(d3.format('.0%')).tickSize(-(h - 26)));
    const tp = tip(el);
    types.forEach((t, i) => {
      const yy = i * (bh + gap) + 2; let acc = 0;
      svg.append('text').attr('class', 'lbl').attr('x', lw - 8).attr('y', yy + bh / 2).attr('dy', 4).attr('text-anchor', 'end').text(t);
      bands.forEach((b, k) => {
        const d = rows.find((q) => q.t === t && q.b === b); if (!d) return;
        svg.append('rect').attr('x', x(acc)).attr('y', yy).attr('width', Math.max(0, x(acc + d.s) - x(acc) - 1)).attr('height', bh).attr('fill', colOf(k))
          .on('mousemove', (ev) => tp.show(ev, `${t} · 난이도 ${b}`, `시도 비중 ${F.pct(d.s)} (${F.int(d.n)}개) · 기대 대비 ${F.pct(d.v)}`)).on('mouseleave', () => tp.hide());
        if (x(acc + d.s) - x(acc) > 34) svg.append('text').attr('x', x(acc + d.s / 2)).attr('y', yy + bh / 2).attr('dy', 4).attr('text-anchor', 'middle').style('fill', k >= bands.length - 2 ? '#fff' : 'var(--color-fg)').style('pointer-events', 'none').text(d3.format('.0%')(d.s));
        acc += d.s;
      });
    });
    el.style.minHeight = h + 'px';
  }

  function draw() {
    fillMarks();
    table('tbl-periods', 'periods', [['period'], ['attempts', 'int'], ['users', 'int'], ['solve_rate', 'pct'], ['first_correct_rate', 'pct'], ['mean_retries', 'f2'], ['solve_before_last7', 'pct'], ['solve_last7', 'pct']]);
    table('tbl-models', 'models', [['key'], ['AUC', 'f3'], ['logloss', 'f3'], ['Brier', 'f3'], ['brier_skill', 'pct']], { hl: (d) => d.key.includes('검증 · 로지스틱') });
    table('tbl-case', 'user_case', [['tag'], ['attempts', 'int'], ['solved', 'int'], ['raw_rate', 'pct'], ['expected', 'f2'], ['vs_expected', 'pct'], ['z', 'f2'], ['judge', null, (d) => (d.judge === '약점' ? 'var(--color-bad)' : d.judge === '강점' ? 'var(--color-ok)' : '')]]);
    table('tbl-case-h2', 'case_h2', [['tag'], ['judge', null, (d) => (d.judge === '약점' ? 'var(--color-bad)' : d.judge === '강점' ? 'var(--color-ok)' : '')], ['n', 'int'], ['ratio', 'pct'], ['n_v', 'int'], ['solved_v', 'int'], ['ratio_v', 'pct']]);
    table('tbl-pred', 'pred_cmp', [['model'], ['AUC', 'f3'], ['logloss', 'f3'], ['Brier', 'f3'], ['logloss_gain_pct', 'gain']], { hl: (d) => d.model.includes('우리') });
    table('tbl-target', 'target_cmp', [['target'], ['target_mean', 'pct'], ['model_auc', 'f3'], ['weak_resid', 'pp'], ['strong_resid', 'pp'], ['hit_ours', 'pct'], ['hit_raw', 'pct'], ['hit_random', 'pct'], ['gain_ours', 'gain'], ['gain_raw', 'gain']]);
    table('tbl-rule', 'rule_cmp', [['rule'], ['diag_attempts', 'int'], ['started_before', 'pct'], ['seen_before', 'pct'], ['weak_resid', 'pp'], ['hit_ours', 'pct'], ['hit_raw', 'pct'], ['hit_random', 'pct']]);
    table('tbl-clusters', 'clusters', [['type'], ['users', 'int'], ['rating', 'int'], ['vs_expected', 'pct'], ['first_correct', 'pct'], ['retries', 'f2'], ['rating_change_h2', 'int']]);
    const vis = (id) => { const s = document.getElementById(id).closest('[data-panel]'); return s && !s.hidden; };
    if (vis('ch-calib')) drawCalib();
    if (vis('ch-raw')) {
      drawBand('ch-raw', 'raw_rate', d3.format('.0%'));
      drawBand('ch-ratio', 'vs_expected', d3.format('.0%'), 1);
      const lg = document.getElementById('lg-case'); lg.innerHTML = '';
      ['약점', '보통', '강점'].forEach((j) => { const s = document.createElement('span'); s.style.setProperty('--sw', JUDGE[j]); s.textContent = j; lg.appendChild(s); });
      drawDiff(); drawDiffHeat();
      hbars('ch-case', 'user_case', {
        map: (d) => ({ label: d.tag, v: num(d.vs_expected), color: JUDGE[d.judge] || 'var(--cds-chart-muted)', tipText: `${d.attempts}문제 중 ${d.solved}개 해결 · 기대 대비 ${F.pct(num(d.vs_expected))} · ${d.judge}` }),
        domain: () => [0, Math.max(1.1, d3.max(dash.data('user_case').data, (d) => num(d.vs_expected)) * 1.05)], base: 0, ref: 1, fmt: F.pct, tick: d3.format('.0%'),
      });
    }
    if (vis('ch-judge')) {
      hbars('ch-judge', 'by_judge', {
        map: (d) => ({ label: `상반기 ${d.judge} (${F.int(num(d.pairs))}쌍)`, v: num(d.resid), lo: num(d.lo), hi: num(d.hi), color: JUDGE[d.judge], tipText: `하반기 시도당 ${F.pp(num(d.resid))} (95% ${F.pp(num(d.lo))} ~ ${F.pp(num(d.hi))})` }),
        domain: (rows) => { const a = d3.min(rows, (d) => d.lo), b = d3.max(rows, (d) => d.hi); const p = (b - a) * 0.08; return [Math.min(0, a) - p, Math.max(0, b) + p]; },
        base: 0, ref: 0, fmt: F.pp, tick: (v) => d3.format('+.0f')(v * 100) + '%p',
      });
      hbars('ch-weak', 'weak_cmp', {
        map: (d) => ({ label: d.method, v: num(d.hit_rate), color: d.method.includes('레이팅 보정') ? dash.colors[0] : 'var(--cds-chart-muted)', tipText: `하반기에도 기대 이하 ${F.pct(num(d.hit_rate))} · ${F.int(num(d.pairs))}쌍` }),
        domain: (rows) => [0, Math.min(1, d3.max(rows, (d) => d.v) * 1.15)], base: 0, ref: 0.5, fmt: F.pct, tick: d3.format('.0%'),
      });
    }
    if (vis('ch-heat')) { drawHeat(); drawK(); drawTypeDiff(); }
  }
  window.cfDraw = draw;
  dash.onData(draw);
})();
</script>
'''

DASH_TOKENS = r''':root { --font-anthropic-sans: "Pretendard", "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", system-ui, sans-serif; --font-anthropic-serif: "Noto Serif KR", "AppleMyungjo", "Nanum Myeongjo", Georgia, serif;
  --color-bg: #fbfbfa; --color-fg: #1d1d1b; --color-fg-muted: #6b6a65; --color-panel: #ffffff; --color-border-line: #e4e3de; --color-ok: #1a7f4b; --color-warn: #b26b00; --color-bad: #c4352f;
  --cds-chart-axis: #bdbcb6; --cds-chart-grid: #ecebe7; --cds-chart-reference: #8a8984; --cds-chart-muted: #b9b8b2; --cds-chart-status-critical: #d4452f;
  --cds-surface-popover: #ffffff; --cds-border: #dcdbd5; --cds-shadow-popover: 0 4px 16px rgba(0,0,0,.12); --cds-text-secondary: #6b6a65;
  --cds-font-size-caption: 12px; --cds-font-size-body: 14px; --cds-font-size-heading: 15px; --cds-font-size-title: 28px; --cds-font-weight-medium: 500;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 12px; --cds-gap-lg: 20px; --cds-gap-xl: 32px; --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 16px; --cds-pad-xl: 24px;
  --cds-radius: 8px; --cds-dur-slow: 300ms; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --color-bg: #1a1a19; --color-fg: #f0efec; --color-fg-muted: #a9a8a1; --color-panel: #232322; --color-border-line: #3a3a37; --color-ok: #4fbf85; --color-bad: #ec6b5e;
  --cds-chart-axis: #55544f; --cds-chart-grid: #2f2f2d; --cds-chart-muted: #5f5e59; --cds-chart-status-critical: #ef6a55; --cds-surface-popover: #2a2a28; --cds-border: #44443f; --cds-text-secondary: #a9a8a1; color-scheme: dark; } }
body { margin: 0; background: var(--color-bg); color: var(--color-fg); font-family: var(--font-anthropic-sans); }
#dash-root { max-width: 1200px; margin: 0 auto; padding: 24px max(16px, env(safe-area-inset-right, 0px)) 48px max(16px, env(safe-area-inset-left, 0px)); }
.cf-foot { margin-top: 32px; font-size: 12px; color: var(--color-fg-muted); }
[hidden] { display: none !important; }'''

DASH_SHIM = r'''window.dash = (function () {
  const D = window.DASH_DATA || {};
  return {
    data(id) { const rows = D[id]; return rows ? { status: 'ok', data: rows, columns: Object.keys(rows[0] || {}), refreshing: false, meta: [] } : { status: 'missing', data: [], columns: [] }; },
    onData(fn) { fn(); let t; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(fn, 150); }); },
    colors: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
    setLink(name) { try { history.replaceState(null, '', '#' + name); } catch (e) {} },
    refresh() {}, params() { return {}; }, setParams() {}, resetParams() {},
  };
})();'''


def write_dashboard(path):
    try:
        import urllib.request
        d3_js = urllib.request.urlopen("https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js", timeout=60).read().decode("utf-8")
        d3_tag = "<script>" + d3_js + "</script>"          # 인터넷 없이 열리도록 d3를 파일 안에 넣음
    except Exception:
        d3_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'
    data_js = json.dumps(dash_datasets(), ensure_ascii=False).replace("</", "<\\/")
    eu = RESULTS["example_user"]
    page = DASH_PAGE.replace('<h3 id="case-title">실제 사용자 사례</h3>',
                             f'<h3 id="case-title">실제 사용자 사례: {eu["handle"]} (최근 레이팅 {eu["rating"]:.0f})</h3>')
    foot = (f'<p class="cf-foot">데이터: usersCodeforcesSubmissionsEnd2024 (제출 {RESULTS["n_submissions"]:,}건) · '
            f'정답값: {TARGET} · 생성: {time.strftime("%Y-%m-%d %H:%M")} · 1_분석코드_Colab용.ipynb 10절</p>')
    html = ("<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Codeforces 풀이 진단 대시보드</title><style>" + DASH_TOKENS + "</style>" + d3_tag +
            "<script>window.DASH_DATA = " + data_js + ";</script><script>" + DASH_SHIM + "</script></head>"
            "<body><div id=\"dash-root\">" + page + foot + "</div></body></html>")
    Path(path).write_text(html, encoding="utf-8")
    return len(html)


size = write_dashboard(OUT / "결과_대시보드.html")
print(f"결과_대시보드.html 저장 ({size/1024:.0f} KB) → 브라우저에서 열면 됩니다")

# %% [markdown]
# ## 11. 결과 한 번에 내려받기
# `분석결과` 폴더 전체를 `분석결과.zip`으로 묶습니다. Colab에서는 자동으로 내려받기가 시작됩니다.
# 실행 결과가 담긴 노트북은 Colab 메뉴 **파일 → 다운로드 → .ipynb 다운로드**로 저장하세요.

# %%
import shutil
zip_path = shutil.make_archive(str(BASE / "분석결과"), "zip", root_dir=OUT)
print("묶음 파일:", zip_path, f"({Path(zip_path).stat().st_size/1e6:.1f} MB)")
if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
