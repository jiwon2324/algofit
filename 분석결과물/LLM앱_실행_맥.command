#!/bin/bash
# 알고핏(AlgoFit) 실행 (맥) — 더블클릭하면 필요한 패키지를 확인·설치하고 브라우저에서 앱을 엽니다.
cd "$(dirname "$0")" || exit 1
APP_DIR="$(ls -d 6_LLM* 2>/dev/null | head -1)"
[ -n "$APP_DIR" ] && cd "$APP_DIR"
[ -f app.py ] || { echo "app.py를 찾지 못했어요. 이 파일을 졸업작품 폴더(또는 앱 폴더) 안에 두고 실행하세요."; read -r -p "엔터를 누르면 닫혀요"; exit 1; }

echo "=== 알고핏(AlgoFit)를 준비하고 있어요 ==="
PY="$(command -v python3)"
if [ -z "$PY" ] || ! "$PY" -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
  echo "파이썬 3.9 이상이 필요해요. https://www.python.org/downloads/ 에서 설치한 뒤 다시 더블클릭하세요."
  open "https://www.python.org/downloads/" 2>/dev/null
  read -r -p "엔터를 누르면 닫혀요"; exit 1
fi

if ! "$PY" -c "import streamlit, openai, dotenv, pandas, docx, altair" 2>/dev/null; then
  echo "처음 실행이라 필요한 패키지를 설치해요 (1~3분 걸려요)..."
  "$PY" -m pip install --user -r requirements.txt \
    || "$PY" -m pip install -r requirements.txt \
    || "$PY" -m pip install --user --break-system-packages -r requirements.txt
  "$PY" -c "import streamlit, openai, dotenv, pandas, docx, altair" 2>/dev/null \
    || { echo "패키지 설치에 실패했어요. 위 오류 메시지를 캡처해서 보내 주세요."; read -r -p "엔터를 누르면 닫혀요"; exit 1; }
fi

# 스트림릿이 처음에 이메일을 물어보며 멈추지 않게 미리 빈 설정을 만들어 둠
mkdir -p "$HOME/.streamlit"
[ -f "$HOME/.streamlit/credentials.toml" ] || printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"

echo ""
echo "브라우저에 앱이 열려요. 다 쓴 뒤에는 이 창을 닫으면 앱이 꺼져요."
"$PY" -m streamlit run app.py
