@echo off
REM Double-click to try the project with FAKE data (refuses to overwrite real data).
python -m pip install -r requirements.txt
python -m crypto_sentiment.demo
if errorlevel 1 pause & exit /b 1
python -m crypto_sentiment.train --full
python -m streamlit run app/dashboard.py
pause
