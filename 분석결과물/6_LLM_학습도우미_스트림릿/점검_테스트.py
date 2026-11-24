"""학습도우미 앱 자동 점검 — 실제 OpenAI를 부르지 않고(비용 0원), 정해 둔 답을 돌려주는 시험용 대역(mock)으로
AI가 이상한 답을 보내거나 연결이 끊기는 경우까지 일부러 만들어 앱이 멈추지 않는지 확인합니다.
실행: 이 폴더에서  python3 점검_테스트.py        (빠르게: python3 점검_테스트.py quick)
"""
import os, sys, json, random, time, traceback, importlib.util, types, io
os.environ["OPENAI_API_KEY"] = "sk-fake-for-test"          # .env 보다 먼저 잡혀서 실제 키를 쓰지 않음
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import tempfile
os.environ["THREAD_DIR"] = tempfile.mkdtemp(prefix="threads_")   # 문제별 대화 저장은 임시 폴더에 (실제 기록 건드리지 않음)

import openai
from streamlit.testing.v1 import AppTest

MODE = {"diag": "good", "help": "steps", "raise": False}
CALLS = []


def _msg(text):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


DIAG = {
    "good": {"summary": "요약", "guidance": {"what": "무엇", "level": "수준", "how": "방법"}, "weak_points": ["a"], "strong_points": ["b"], "habits": ["c"],
             "plan": [{"week": "1주차", "focus": "DP", "difficulty": "쉬움(800~1199)", "practice": "연습"},
                      {"week": 2, "focus": "DP", "difficulty": "보통", "practice": ["x", "y"]}]},
    "missing": {"summary": "요약만"},
    "guidance_str": {"summary": "s", "guidance": "문자열 지도 방향"},
    "weird": {"summary": ["리스트 요약"], "weak_points": "문자열 하나", "strong_points": None,
              "habits": [{"x": 1}], "plan": {"week": "1주차", "focus": "DP"}},
    "plan_strings": {"summary": "s", "weak_points": [], "strong_points": [], "habits": [], "plan": ["1주차: DP", "2주차: 그리디"]},
    "html": {"summary": "<script>alert(1)</script><b>굵게</b>", "weak_points": ["<i>x</i>"], "strong_points": [], "habits": [],
             "plan": [{"week": "1주차", "focus": "<b>DP</b>", "difficulty": "-", "practice": "-"}]},
}
HELP = {
    "steps": "안녕하세요!\n\n1단계: **처음** 보세요.\n2단계: 다음.\n3단계: 마지막.\n\n---\n화이팅",
    "plain": "그냥 설명입니다. 단계 없음.",
    "empty": "",
    "math": "수식 \\[ a^2 + b^2 \\] 과 \\( x \\)",
}


class FakeCompletions:
    def create(self, model, messages, response_format=None, **kw):
        CALLS.append(model)
        if MODE["raise"]:
            raise RuntimeError("network down (test)")
        if response_format:
            v = DIAG[MODE["diag"]]
            return _msg(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        if isinstance(messages[-1]["content"], list):        # 이미지
            return _msg("문제: 두 수를 더하라")
        return _msg(HELP[MODE["help"]])


class FakeAudio:
    class transcriptions:
        @staticmethod
        def create(**kw):
            return "음성 질문입니다"


class FakeOpenAI:
    def __init__(self, api_key=None, **kw):
        self.chat = types.SimpleNamespace(completions=FakeCompletions())
        self.audio = FakeAudio()


openai.OpenAI = FakeOpenAI

FAIL = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAIL.append(name + (f": {detail}" if detail else ""))


def errs(at):
    return [e.value for e in at.exception]


def new_app():
    return AppTest.from_file(os.path.abspath("app.py"), default_timeout=120)


def hid(at):
    return next(w for w in at.text_input if w.label == "Codeforces 아이디")


def nm_in(at):
    return next(w for w in at.text_input if w.label == "이름 (선택)")


def btn(at, label):
    for b in at.button:
        if b.label.startswith(label):
            return b
    raise KeyError(label)


# ---------------------------------------------------------------- A. 함수 단위: 전체 사용자
spec = importlib.util.spec_from_file_location("app_mod", "학습도우미_app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
data_path = next(p for p in app.DATA_CANDIDATES if p.exists())
profiles = [json.loads(l) for l in open(data_path, encoding="utf-8")]
quick = len(sys.argv) > 1
print(f"[A] 전체 사용자 {len(profiles):,}명 함수 점검")
t0 = time.time()
bad = []
for p in profiles:
    try:
        app.diagnosis_text(p); app.notable_habits(p); app.level_text(p)
        for t in p["weak_tags"] + p["strong_tags"]:
            app.expected_count(t); app.tag_ko(t["tag"])
        for b in p["difficulty_bands"]:
            app.diff_name(b["band"])
    except Exception as e:
        bad.append((p["handle"], repr(e)))
check("진단 요약 만들기 (전체 사용자)", not bad, str(bad[:3]))
bad = []
for p in profiles:
    for c in [None] + [x["category"] for x in p["categories"]]:
        for r in [None, 800, 3500]:
            amt, why = app.help_amount_rule(p, c, r)
            if amt not in app.HELP_AMOUNTS or not why:
                bad.append(p["handle"])
check("도움 정도 규칙 (전체 사용자 × 분야 × 난이도)", not bad, str(bad[:3]))
u = {p["handle"]: p for p in profiles}
check("실력 부족 + 약점 분야 + 어려운 문제 → 많이", app.help_amount_rule(u["user1610"], "수학", 1500)[0] == "많이")
check("관계없는 질문 안내 뒤에 힌트가 붙어도 안내 문장만 남김", app.apply_guard(app.GUARD_OFF_TOPIC + "\n\n1단계: 힌트") == app.GUARD_OFF_TOPIC)
check("레이팅 이상 + 강점 분야 → 조금씩", app.help_amount_rule(u["user12313"], "탐색")[0] == "조금씩")
check("레이팅 이상 + 약점 분야 → 보통", app.help_amount_rule(u["user12313"], "DP")[0] == "보통")
no_desc = [p["handle"] for p in profiles if not p.get("learning_type_desc")]
no_type = [p["handle"] for p in profiles if not p.get("learning_type")]
unk_tags = sorted({t["tag"] for p in profiles for t in p["weak_tags"] + p["strong_tags"]} - set(app.TAG_KO) - set(app.TAG_DESC))
unk_band = sorted({b["band"] for p in profiles for b in p["difficulty_bands"]} - set(app.DIFF_NAME))
unk_level = sorted({p["rating_band"] for p in profiles} - set(app.LEVEL_NAME))
zero_vs = [p["handle"] for p in profiles for t in p["weak_tags"] + p["strong_tags"] if not t["vs_expected"]]
print(f"  참고: 학습유형 없음 {len(no_type)}명, 유형설명 없음 {len(no_desc)}명, 기대대비 0 {len(zero_vs)}건")
check("모든 태그에 한글 이름 있음", not unk_tags, str(unk_tags))
check("모든 난이도 구간에 이름 있음", not unk_band, str(unk_band))
check("모든 실력 구간에 단계 이름 있음", not unk_level, str(unk_level))

sample = random.Random(0).sample(profiles, 60 if quick else 400)
edge = [p for p in profiles if not p["weak_tags"]][:5] + [p for p in profiles if not p["strong_tags"]][:5] \
     + [p for p in profiles if len(p["weak_tags"]) >= 4][:5] + [p for p in profiles if not p.get("learning_type")][:3] \
     + [p for p in profiles if p.get("effective_rating") is None][:5]
bad = []
for p in sample + edge:
    for d in [DIAG["good"], {"summary": ""}, {}]:
        try:
            dd = app.normalize_diagnosis(d) if hasattr(app, "normalize_diagnosis") else d
            app.build_report_docx(p, dd, random.choice(["", "김코딩", "a/b:c*?", "<b>x</b>"])).read()
        except Exception as e:
            bad.append((p["handle"], repr(e)[:120]))
check(f"Word 리포트 만들기 ({len(sample + edge)}명 × 3가지 AI 결과)", not bad, str(bad[:3]))
print(f"  ({time.time() - t0:.0f}초)")

# ---------------------------------------------------------------- B. 화면: 여러 사용자
print("[B] 화면 점검")
at = new_app().run()
check("첫 화면 오류 없음", not errs(at), str(errs(at)))
check("첫 화면에 미리 만든 AI 결과 보임", any("AI 코치 한마디" in m.value for m in at.markdown))
check("첫 화면에 Word 저장 버튼 있음", len(at.get("download_button")) == 1)

users = [p["handle"] for p in random.Random(1).sample(profiles, 8 if quick else 40)] + [p["handle"] for p in edge]
bad = []
for h in users:
    hid(at).set_value(h).run()
    if errs(at):
        bad.append((h, errs(at)[0][:120]))
check(f"사용자 바꾸기 ({len(users)}명, 약점·강점 없는 사람·평소 실력 계산 불가 포함)", not bad, str(bad[:3]))

for v, label in [("nobody", "없는 아이디"), ("", "빈 아이디"), ("  user12313  ", "앞뒤 공백 아이디"), ("USER12313", "대문자 아이디")]:
    hid(at).set_value(v).run()
    check(f"{label} 입력 시 오류 없음", not errs(at), str(errs(at)))
at = new_app().run()                                    # 새 화면에서 확인
hid(at).set_value("  user12313  ").run()
check("공백 붙은 아이디도 찾음", any("user12313님" in m.value for m in at.markdown))

for nm in ["<script>alert(1)</script>", "a/b:c*?\"<>|", "김코딩 😀", "x" * 200]:
    nm_in(at).set_value(nm).run()
    hero = next((m.value for m in at.markdown if "학습 리포트</h2>" in m.value), "")
    check(f"이름 '{nm[:20]}' 오류 없음", not errs(at), str(errs(at)))
    if "<script>" in nm:
        check("이름에 넣은 HTML이 그대로 실행되지 않음", "<script>" not in hero, hero[:80])
    fn = app.safe_filename(nm)
    check(f"저장 파일 이름에 쓸 수 없는 글자 없음 ({nm[:12]})", fn and not any(c in fn for c in '/\\:*?"<>|'), fn)
nm_in(at).set_value("").run()

# ---------------------------------------------------------------- C. AI 코치 (가짜 AI)
print("[C] AI 코치 응답 점검")
for mode in ["good", "missing", "weird", "plan_strings", "html"]:
    MODE.update({"diag": mode, "raise": False})
    at = new_app().run()
    btn(at, "AI 코치에게").click().run()
    check(f"AI 응답 '{mode}' 오류 없음", not errs(at), str(errs(at))[:200])
    if mode == "html":
        coach = next((m.value for m in at.markdown if "AI 코치 한마디" in m.value), "")
        check("AI가 쓴 HTML이 그대로 실행되지 않음", "<script>" not in coach, coach[:100])
    if not errs(at) and at.get("download_button"):
        check(f"AI 응답 '{mode}' 뒤 Word 저장 버튼 있음", True)
MODE.update({"diag": "good", "raise": True})
at = new_app().run()
btn(at, "AI 코치에게").click().run()
check("AI 연결 실패 시 앱이 멈추지 않고 안내", not errs(at) and any("문제가 생겼어요" in e.value for e in at.error), str(errs(at)))
MODE["raise"] = False

# ---------------------------------------------------------------- D. 문제 풀이 도움 (가짜 AI)
print("[D] 문제 풀이 도움 점검 (문제별 대화)")
from pathlib import Path
at = new_app().run()
th = at.session_state["threads"]["user12313"]
check("예시 문제 대화가 목록에 있음", not errs(at) and len(th) >= 1 and th[0]["messages"], str(errs(at)))
btn(at, "새로 만들기").click().run()
btn(at, "캡처에서 문제 읽기").click().run()
check("이미지 없이 캡처 읽기 → 안내만", not errs(at) and any("올려 주세요" in w.value for w in at.warning), str(errs(at)))
btn(at, "저장하고 질문 시작").click().run()
check("빈 칸으로 저장 → 안내만", not errs(at) and any("적어 주세요" in w.value for w in at.warning), str(errs(at)))
n_th = len(at.session_state["threads"]["user12313"])
next(w for w in at.text_input if w.label == "문제 제목").set_value("테스트 DP 문제").run()
next(w for w in at.selectbox if w.label == "문제 분야").set_value("DP").run()
next(w for w in at.text_area if w.label == "문제").set_value("두 수를 더하는 문제").run()
btn(at, "저장하고 질문 시작").click().run()
th = at.session_state["threads"]["user12313"]
cur = th[-1]
at.session_state["current"]["user12313"] = cur["id"]
check("새 문제 저장 → 목록에 추가되고 그 문제가 열림", not errs(at) and len(th) == n_th + 1 and at.session_state["current"]["user12313"] == cur["id"], str(errs(at)))
check("저장 파일이 생김 (앱을 다시 켜도 남음)", Path(os.environ["THREAD_DIR"], "user12313.json").exists())
check("문제 분야(DP)에 맞춰 추천 도움 정도와 이유가 보임", any("DP는 보완이 필요해 보이는 분야" in c.value for c in at.caption))
at = new_app().run()                                    # 새 화면으로 다시 열기 → 왼쪽 목록에서 방금 저장한 문제 누르기
btn(at, "테스트 DP 문제").click().run()
check("다시 열어도 방금 저장한 문제가 목록에 있고 열림", not errs(at) and any("테스트 DP 문제" in m.value for m in at.markdown), str(errs(at)))
for mode in ["steps", "plain", "empty", "math"]:
    MODE["help"] = mode
    at.get("button_group")[0].set_value("많이").run()
    at.chat_input[0].set_value(f"질문 {mode}").run()
    check(f"힌트 응답 '{mode}' 오류 없음", not errs(at), str(errs(at))[:200])
cur = at.session_state["threads"]["user12313"][-1]
check("질문과 답이 이 문제 대화에 쌓임", len(cur["messages"]) == 4, str(len(cur["messages"])))
MODE["help"] = "steps"
at.get("button_group")[0].set_value("조금씩").run()
at.chat_input[0].set_value("조금씩 질문").run()
nb = [b for b in at.button if b.label.startswith("다음 힌트 보기")]
check("'조금씩'이면 힌트가 하나씩 열림 (다음 힌트 버튼)", not errs(at) and len(nb) >= 1, str(errs(at))[:200])
if nb:
    lab = nb[-1].label; nb[-1].click().run()
    nb2 = [b for b in at.button if b.label.startswith("다음 힌트 보기")]
    check("다음 힌트 누르면 한 단계 더 보임", not errs(at) and (not nb2 or nb2[-1].label != lab), lab)
MODE["raise"] = True
at.chat_input[0].set_value("실패 질문").run()
check("힌트 AI 연결 실패 시 안내", not errs(at) and any("문제가 생겼어요" in e.value for e in at.error), str(errs(at)))
MODE["raise"] = False
next(w for w in at.text_area if w.label == "내 코드").set_value("고친 코드").run()
btn(at, "코드 저장").click().run()
check("코드 고쳐서 저장", not errs(at) and at.session_state["threads"]["user12313"][-1]["code"] == "고친 코드", str(errs(at)))
first = at.session_state["threads"]["user12313"][0]
btn(at, first["title"]).click().run()
check("목록에서 다른 문제 누르면 그 대화로 바뀜", not errs(at) and at.session_state["current"]["user12313"] == first["id"], str(errs(at)))
at2 = new_app().run()
check("앱을 새로 켜도 저장한 문제가 남아 있음", len(at2.session_state["threads"]["user12313"]) == n_th + 1)
btn(at2, "테스트 DP 문제").click().run()
tid = at2.session_state["threads"]["user12313"][-1]["id"]
next(w for w in at2.text_input if w.key == f"rn_{tid}").set_value("이름 바꾼 문제").run()
next(b for b in at2.button if b.key == f"rns_{tid}").click().run()
check("⋯ 메뉴: 이름 바꾸기", not errs(at2) and at2.session_state["threads"]["user12313"][-1]["title"] == "이름 바꾼 문제", str(errs(at2)))
next(b for b in at2.button if b.key == f"clr_{tid}").click().run()
check("⋯ 메뉴: 대화 내용 지우기", not errs(at2) and at2.session_state["threads"]["user12313"][-1]["messages"] == [], str(errs(at2)))
MODE["help"] = "steps"
btn(at2, "이름 바꾼 문제").click().run()
btn(at2, "어디서부터 생각해야").click().run()
check("추천 질문 누르면 바로 질문됨", not errs(at2) and len(at2.session_state["threads"]["user12313"][-1]["messages"]) == 1, str(errs(at2)))
MODE["help"] = "plain"
btn(at2, "다시 답변 받기").click().run()
m = at2.session_state["threads"]["user12313"][-1]["messages"]
check("다시 답변 받기 (같은 질문으로 새 답변, 개수 그대로)", not errs(at2) and len(m) == 1 and m[0]["answer"] == HELP["plain"], str(errs(at2)))
MODE["raise"] = True
btn(at2, "다시 답변 받기").click().run()
check("다시 답변 받기 실패해도 원래 답변은 남음", not errs(at2) and len(at2.session_state["threads"]["user12313"][-1]["messages"]) == 1, str(errs(at2)))
MODE["raise"] = False
next(b for b in at2.button if b.key == f"slv_{tid}").click().run()
check("해결로 표시 → 목록에 ✓", not errs(at2) and at2.session_state["threads"]["user12313"][-1].get("solved")
      and any(b.label.startswith("✓ 이름 바꾼 문제") for b in at2.button), str(errs(at2)))
btn(at2, "✓ 이름 바꾼 문제").click().run()
check("대화 내보내기 버튼 있음", not errs(at2) and any(d.label == "대화 내보내기" for d in at2.get("download_button")), str(errs(at2)))
md = app.thread_markdown(at2.session_state["threads"]["user12313"][-1])
check("내보낸 노트에 문제·질문·답변이 들어감", "## 문제" in md and "## 질문 1" in md and "### 답변" in md, md[:80])
next(b for b in at2.button if b.key == f"del_{tid}").click().run()
check("문제 지우기", not errs(at2) and len(at2.session_state["threads"]["user12313"]) == n_th, str(errs(at2)))
hid(at2).set_value("user14684").run()
check("예시 대화가 없는 사용자는 빈 목록 + 새 문제 화면", not errs(at2) and any("어떤 문제에서 막혔나요" in m.value for m in at2.markdown), str(errs(at2)))

# ---------------------------------------------------------------- E. API 키 없음
print("[E] API 키 없음 점검")
os.environ["OPENAI_API_KEY"] = ""
at = new_app().run()
check("키 없이 첫 화면 오류 없음 (미리 만든 결과 보임)", not errs(at) and any("AI 코치 한마디" in m.value for m in at.markdown))
btn(at, "AI 코치에게").click().run()
check("키 없이 AI 버튼 → 안내만", not errs(at) and any("사용할 수 없어요" in w.value for w in at.warning), str(errs(at)))
btn(at, "인접하지 않은 원소").click().run()
at.chat_input[0].set_value("질문").run()
check("키 없이 힌트 질문 → 안내만", not errs(at) and any("사용할 수 없어요" in w.value for w in at.warning), str(errs(at)))
os.environ["OPENAI_API_KEY"] = "sk-fake-for-test"

print(f"\n시험용 AI 대역 호출 {len(CALLS)}회 (실제 API 호출 0회)")
print("결과:", "모두 통과" if not FAIL else f"실패 {len(FAIL)}건")
for f in FAIL:
    print(" -", f)
