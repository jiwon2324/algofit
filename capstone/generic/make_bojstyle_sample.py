"""
시험용 '백준 형식' 제출 기록 만들기 — 진짜 백준 데이터가 아니라, Codeforces 제출 기록(B 기간, 학생 1,500명)을
열 이름 · 결과 이름 · 알고리즘 분류 · 문제 번호 · 아이디를 모두 백준식으로 바꾸고 난이도·레이팅은 지운 것.
목적: 형식이 전혀 다른 데이터도 대응표(platforms/baekjoon_style.json)만으로 그대로 진단되는지 보여 주기
실행: cd capstone && .venv/bin/python generic/make_bojstyle_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "sample"; OUT.mkdir(exist_ok=True)
B0, B1 = "2024-01-01", "2024-07-01"

sub = pd.read_csv(ROOT / "output" / "submissions_clean.csv", usecols=["handle", "problem_id", "verdict", "time", "categories"], parse_dates=["time"])
sub = sub[(sub["time"] >= B0) & (sub["time"] < B1)]
users = pd.read_csv(ROOT / "output" / "step3_diagnosis_users.csv")["handle"]
pick = pd.Series(users.sample(1500, random_state=7).to_numpy())
sub = sub[sub["handle"].isin(set(pick))].copy()

VERDICT = {"OK": "맞았습니다!!", "WRONG_ANSWER": "틀렸습니다", "TIME_LIMIT_EXCEEDED": "시간 초과", "MEMORY_LIMIT_EXCEEDED": "메모리 초과",
           "RUNTIME_ERROR": "런타임 에러 (Segfault)", "COMPILATION_ERROR": "컴파일 에러", "PRESENTATION_ERROR": "출력 형식이 잘못되었습니다"}
TAG = {"수학": "수학", "그래프": "그래프 이론", "DP": "다이나믹 프로그래밍", "자료구조": "자료 구조", "탐색": "이분 탐색",
       "그리디·정렬": "그리디 알고리즘", "구현·구성": "구현", "문자열": "문자열", "기하": "기하학", "기타": "게임 이론"}
pid = {p: str(1000 + i) for i, p in enumerate(sorted(sub["problem_id"].unique()))}
uid = {h: f"학생{i + 1:04d}" for i, h in enumerate(pick)}
boj = pd.DataFrame({
    "아이디": sub["handle"].map(uid),
    "문제번호": sub["problem_id"].map(pid),
    "제출시각": sub["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    "결과": sub["verdict"].map(VERDICT).fillna("출력 초과"),
    "알고리즘 분류": sub["categories"].fillna("").map(lambda s: ", ".join(TAG[c] for c in s.split(";") if c in TAG)),
})
boj.to_csv(OUT / "예시_백준형식_제출기록.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"원래 handle": list(uid), "아이디": list(uid.values())}).to_csv(OUT / "예시_아이디_대응(검증용).csv", index=False, encoding="utf-8-sig")
print(f"{len(boj):,}개 제출, 학생 {boj['아이디'].nunique():,}명, 문제 {boj['문제번호'].nunique():,}개 → {OUT / '예시_백준형식_제출기록.csv'}")
print(boj.head(5).to_string(index=False))
