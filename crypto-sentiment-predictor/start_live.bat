@echo off
REM Double-click for the REAL-TIME version: collector in one window, dashboard in another.
python -m pip install -r requirements.txt
start "Collector (keep open)" cmd /k python -m crypto_sentiment.pipeline --loop --interval 300 --retrain-every 12
python -m streamlit run app/dashboard.py
pause
