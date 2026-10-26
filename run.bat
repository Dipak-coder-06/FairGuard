@echo off
cd /d "%~dp0"
set GIT_PYTHON_REFRESH=quiet
call venv\Scripts\activate.bat
streamlit run app.py
