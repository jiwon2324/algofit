"""
새 데이터셋(다른 채점 사이트)에 진단 알고리즘 적용하기 — 재사용 2단계

준비물
  1) 제출 기록 CSV: 학생 · 문제 · 제출시각 · 결과 · 분야(알고리즘 분류) 5개 열 (문제 난이도·학생 레이팅은 있으면 쓰고 없어도 됨)
  2) 플랫폼 대응표: platforms/ 폴더의 json을 복사해서 열 이름 · 결과 이름 · 분야 태그만 그 사이트에 맞게 바꾸기

실행 예시
  .venv/bin/python generic/run_generic.py --platform baekjoon_style --data generic/sample/예시_백준형식_제출기록.csv --out generic/output/백준형식_예시
  (기간을 정하려면 --start 2024-01-01 --end 2024-07-01)

결과 (--out 폴더)
  진단_사용자.csv  진단_분야.csv  학습유형.csv  문제난이도.csv(이 사이트의 저장된 난이도 → 새 학생은 다시 추정 없이 진단)  보고서.txt
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

import generic_engine as ge

ap = argparse.ArgumentParser()
ap.add_argument("--platform", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start"); ap.add_argument("--end")
ap.add_argument("--min-problems", type=int, default=None, help="진단 대상 최소 문제 수 (기본: 저장된 팀 기준 30)")
ap.add_argument("--difficulty", help="이미 저장한 이 사이트의 문제난이도.csv (주면 다시 추정하지 않고 새 학생만 진단)")
args = ap.parse_args()

t0 = time.time()
S = ge.engine.load_settings()
cfg = ge.load_platform(args.platform)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rep = [f"[플랫폼] {cfg['이름']}  /  데이터 {args.data}"]

raw = pd.read_csv(args.data)
sub, srep = ge.standardize(raw, cfg)
rep.append(f"[1] 공통 형식 변환: {srep}")
end = args.end or (sub["time"].max() + pd.Timedelta(seconds=1))
up, sub_cut = ge.make_user_problem(sub, args.start, end, S["기간"]["관찰부족_제외_일수"])
min_p = args.min_problems or S["최소_문제수"]["진단_대상"]
cnt = up.groupby("handle").size()
diag = cnt[cnt >= min_p].index
rep.append(f"[2] 사람×문제 {len(up):,}줄 (첫 정답에서 자르기, 기간 끝 {S['기간']['관찰부족_제외_일수']}일 제외), 진단 대상 {len(diag):,}명 ({min_p}문제 이상)")

if args.difficulty:                                  # 재사용 ①: 저장된 난이도로 새 학생 진단 (다시 학습하지 않음)
    diff = pd.read_csv(args.difficulty, dtype={"problem_id": str})
    rep.append(f"[3] 저장된 문제 난이도 사용: {args.difficulty}")
else:                                                # 재사용 ②: 새 사이트는 풀이 기록으로 난이도 추정 후 저장
    diff, _ = ge.estimate_difficulty(up, S)
    diff.to_csv(out / "문제난이도.csv", index=False, encoding="utf-8-sig")
rep.append(f"[3] 문제 {len(diff):,}개 난이도 (도전 학생 {ge.MIN_PROBLEM_ATTEMPTS}명 이상 {int((diff['도전 학생 수'] >= ge.MIN_PROBLEM_ATTEMPTS).sum()):,}개를 진단에 사용)")

D, C, upd = ge.diagnose(up[up["handle"].isin(diag)], diff, S)
saved = None
if args.difficulty:                                  # 저장된 실력 구간·표준화 값도 같이 사용
    saved = json.loads((Path(args.difficulty).parent / "학습유형_구간별_표준화.json").read_text(encoding="utf-8"))
X, stats = ge.learning_types(upd, sub_cut, D, S, saved=saved)
D["학습 유형"] = X["학습 유형"].reindex(D.index)
D.to_csv(out / "진단_사용자.csv", encoding="utf-8-sig")
C.to_csv(out / "진단_분야.csv", index=False, encoding="utf-8-sig")
X.to_csv(out / "학습유형.csv", encoding="utf-8-sig")
if not saved:
    (out / "학습유형_구간별_표준화.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

rep.append(f"[4] 진단 {len(D):,}명: 분야 판정 " + ", ".join(f"{k} {v:,}" for k, v in C["judge"].value_counts().items()))
rep.append("    약점이 1개 이상인 학생 " + f"{(D['약점 카테고리'] != '').sum():,}명, 강점이 1개 이상인 학생 {(D['강점 카테고리'] != '').sum():,}명")
rep.append("[5] 학습 유형: " + ", ".join(f"{k} {v:,}명" for k, v in X["학습 유형"].value_counts().items()))
rep.append(f"({time.time() - t0:.0f}초)")
(out / "보고서.txt").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
