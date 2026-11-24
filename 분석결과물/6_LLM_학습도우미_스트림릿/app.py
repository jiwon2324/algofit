# 한글 파일 이름을 입력하기 어려울 때 쓰는 실행 파일
# 실행: streamlit run app.py   →  같은 폴더의 학습도우미_app.py를 그대로 실행합니다.
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "학습도우미_app.py"), run_name="__main__")
