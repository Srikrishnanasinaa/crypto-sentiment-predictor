@echo off
REM One click: download ~14 days of history, train, show results (about 10 minutes).
REM Close start_live.bat windows first so two programs do not write the same files.
python -m pip install -r requirements.txt
python -m crypto_sentiment.backfill --days 14
if errorlevel 1 pause & exit /b 1
python -m crypto_sentiment.train --full
python -m streamlit run app/dashboard.py
pause
