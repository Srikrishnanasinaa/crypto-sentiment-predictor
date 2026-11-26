.PHONY: install test demo live train status
install:  ; python -m pip install -r requirements-dev.txt
test:     ; python -m pytest -q
demo:     ; python -m crypto_sentiment.demo && python -m crypto_sentiment.train --full && python -m streamlit run app/dashboard.py
live:     ; python -m crypto_sentiment.pipeline --loop --retrain-every 12
train:    ; python -m crypto_sentiment.train --full
status:   ; python -m crypto_sentiment.status
