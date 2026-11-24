@echo off
chcp 65001 > nul
title AlgoFit
rem AlgoFit launcher (Windows): double-click to check packages and open the app in the browser.
cd /d "%~dp0"
for /d %%D in (6_LLM*) do cd /d "%%D"
if not exist app.py (
  echo [!] app.py not found. Keep this file inside the project folder.
  pause
  exit /b 1
)

set "PY="
py -3 -c "import sys" > nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys" > nul 2>&1 && set "PY=python"
if not defined PY (
  echo [!] Python 3 is not installed.
  echo     Install it from https://www.python.org/downloads/
  echo     IMPORTANT: check "Add python.exe to PATH" on the first install screen.
  start "" https://www.python.org/downloads/
  pause
  exit /b 1
)

%PY% -c "import streamlit, openai, dotenv, pandas, docx, altair" > nul 2>&1
if errorlevel 1 (
  echo First run: installing packages. This takes 1-3 minutes...
  %PY% -m pip install -r requirements.txt
  %PY% -c "import streamlit, openai, dotenv, pandas, docx, altair" > nul 2>&1
  if errorlevel 1 (
    echo [!] Package install failed. Please send a screenshot of the messages above.
    pause
    exit /b 1
  )
)

rem Skip Streamlit first-run email question
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo.
echo The app opens in your browser. Close this window to stop the app.
%PY% -m streamlit run app.py
pause
