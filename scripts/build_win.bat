@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
pyinstaller surgicalai.spec --clean
endlocal
