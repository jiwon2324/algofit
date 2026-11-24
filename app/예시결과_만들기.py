"""
앱을 켜자마자 보여 줄 '실제 AI 결과'를 미리 만들어 예시결과.json에 저장합니다.
- 발표 사례 6명의 진단 설명·맞춤 지도 방향 6회 + 같은 DP 문제에 대한 힌트 3명(도움 정도 많이·보통·조금씩) 3회 = OpenAI API 9회 호출
- 사례 고르는 규칙: 학습 유형 4개마다 '약점이 1개 이상인 사람 중 진단 기간 문제 수가 가장 많은 사람' 1명씩
  + 진단 그림 사례 2(user1610, 실력 부족) + '레이팅 수준' 판정 중 같은 규칙으로 고른 1명(user9513)
- 실행: python 예시결과_만들기.py   (같은 폴더의 학습도우미_app.py 함수를 그대로 사용)
"""
import json
import time
import importlib.util
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("app", APP_DIR / "학습도우미_app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

from openai import OpenAI

HANDLE = app.DEFAULT_USER
CASES = ["user12313", "user14684", "user4864", "user10548", "user1610", "user9513"]
EXAMPLE_PROBLEM = "길이 n인 배열에서 서로 인접하지 않은 원소들을 골라 합이 최대가 되게 하는 문제 (n <= 2*10^5)"
EXAMPLE_CODE = """dp = [0]*n
for i in range(n):
    dp[i] = max(dp[i-1], dp[i-2] + a[i])
print(dp[n-1])"""
EXAMPLE_QUESTION = "예제는 맞는데 제출하면 틀려요. 왜 그런지 모르겠어요"
EXAMPLE_CATEGORY = "DP"
EXAMPLE_TITLE = "인접하지 않은 원소 최대 합 (예시)"
HELP_CASES = ["user1610", "user12313", "user4864"]   # 같은 문제 → 실력 부족(많이) / DP 약점(보통) / DP 약점 아님·레이팅 이상(조금씩)


def main():
    if not app.key:
        print(".env 파일에 OPENAI_API_KEY가 없습니다.")
        return
    data_path = next(p for p in app.DATA_CANDIDATES if p.exists())
    profiles = {}
    for l in open(data_path, encoding="utf-8"):
        h = json.loads(l)["handle"]
        if h in CASES:
            profiles[h] = json.loads(l)
    p = profiles[HANDLE]
    client = OpenAI(api_key=app.key)

    users = {}
    for i, h in enumerate(CASES, 1):
        print(f"1-{i}) {h} 진단 설명·지도 방향 만드는 중...")
        users[h] = {"diagnosis": app.generate_diagnosis(client, profiles[h])}

    for i, h in enumerate(HELP_CASES, 1):
        level, reasons = app.help_amount_rule(profiles[h], EXAMPLE_CATEGORY)
        print(f"2-{i}) {h} 힌트 만드는 중... (도움 정도: {level})")
        user_prompt = app.help_user_prompt(level, EXAMPLE_PROBLEM, EXAMPLE_CODE, EXAMPLE_QUESTION, EXAMPLE_TITLE)
        answer = app.process_messages([{"role": "system", "content": app.help_system_prompt(profiles[h])},
                                       {"role": "user", "content": user_prompt}], client)
        users[h]["threads"] = [{"id": "example1", "title": EXAMPLE_TITLE, "category": EXAMPLE_CATEGORY, "rating": None,
                                "problem": EXAMPLE_PROBLEM, "code": EXAMPLE_CODE,
                                "messages": [{"level": level, "reasons": reasons, "question": EXAMPLE_QUESTION, "user": user_prompt, "answer": answer}]}]

    result = {
        "model": app.TEXT_MODEL,
        "created_at": time.strftime("%Y-%m-%d"),
        "users": users,
    }
    with open(APP_DIR / "예시결과.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("예시결과.json 저장 완료")


if __name__ == "__main__":
    main()
