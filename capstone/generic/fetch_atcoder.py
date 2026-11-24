"""
실제 다른 플랫폼 데이터 받기 — AtCoder (일본 프로그래밍 대회 사이트)
출처: AtCoder Problems 공개 API (https://kenkoooo.com/atcoder/) — 사용자별 실제 제출 기록, 문제별 난이도 추정값
  - 요청은 1초에 1번 이하 (서비스 안내를 지킴)
  - 최적화 대회(AHC)는 점수제라 정답/오답 판정이 아니므로 제외
  - 문제 난이도(problem-models.json)는 진단에는 쓰지 않고, 우리 Rasch 추정이 맞는지 '검증'에만 쓴다
실행: cd capstone && .venv/bin/python generic/fetch_atcoder.py
결과: generic/sample/atcoder_제출기록.csv, generic/sample/atcoder_문제난이도_검증용.csv
"""
import json
import random
import time
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "sample"; OUT.mkdir(exist_ok=True)
API = "https://kenkoooo.com/atcoder/atcoder-api/v3"
UA = {"User-Agent": "capstone-graduation-project (research, 1 req/s)", "Accept-Encoding": "identity"}
START, END = 0, 1735689600                   # 처음 가입했을 때부터 ~ 2025-01-01 (UTC) — '첫 제출' 규칙을 지키려면 전체 기록이 필요

N_USERS = 400


def get(url):
    time.sleep(1.05)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


# 1) 사용자 고르기: 정답 수 순위 2,000~6,000위에서 무작위 (꾸준히 푸는 사람들)
USERS_FILE = OUT / "atcoder_대상학생.txt"
if USERS_FILE.exists():                              # 처음 고른 학생 그대로 (2024년에 제출이 있는 294명)
    users = USERS_FILE.read_text(encoding="utf-8").split()
else:
    pool = []
    for f in range(2000, 6000, 1000):                # 한 번에 1,000명까지만 받을 수 있음
        pool += get(f"{API}/ac_ranking?from={f}&to={f + 1000}")
    users = sorted(u["user_id"] for u in random.Random(42).sample(pool, N_USERS))
print(f"사용자 {len(users)}명 선택", flush=True)

rows = []
for i, u in enumerate(users, 1):
    since = START
    while True:
        try:
            batch = get(f"{API}/user/submissions?user={u}&from_second={since}")
        except Exception as e:
            print("  다시 시도:", u, e, flush=True); time.sleep(5); continue
        batch = [b for b in batch if b["epoch_second"] < END]
        rows += batch
        if len(batch) < 500:
            break
        since = max(b["epoch_second"] for b in batch) + 1
    if i % 25 == 0:
        print(f"  {i}/{len(users)}명, 제출 {len(rows):,}개", flush=True)

df = pd.DataFrame(rows).drop_duplicates("id")       # 같은 제출(같은 제출 번호)이 두 번 받아진 경우 제거
df = df[~df["contest_id"].str.startswith("ahc")]                         # 최적화(점수제) 대회 제외
df["submitted_at"] = pd.to_datetime(df["epoch_second"], unit="s")
df[["id", "user_id", "problem_id", "contest_id", "submitted_at", "result", "language"]].to_csv(OUT / "atcoder_제출기록.csv", index=False, encoding="utf-8-sig")
print(f"저장: 제출 {len(df):,}개, 사용자 {df['user_id'].nunique()}명, 문제 {df['problem_id'].nunique():,}개", flush=True)

models = get("https://kenkoooo.com/atcoder/resources/problem-models.json")
pm = pd.DataFrame([{"problem_id": k, "difficulty": v.get("difficulty"), "is_experimental": v.get("is_experimental")} for k, v in models.items()])
pm.to_csv(OUT / "atcoder_문제난이도_검증용.csv", index=False, encoding="utf-8-sig")
print("끝", flush=True)
