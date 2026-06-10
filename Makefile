PY := venv/bin/python

.PHONY: setup snapshot profile test backtest-fast backtest backtest-tuned forecast tune-prophet tune-xgboost tune-sarima tune-monte_carlo tune-ets

setup:
	python3.12 -m venv venv
	venv/bin/pip install --no-cache-dir -r requirements.txt -e .

snapshot:
	$(PY) scripts/make_snapshot.py

profile:
	$(PY) scripts/profile_data.py

test:
	$(PY) -m pytest -q

backtest-fast:
	$(PY) scripts/run_backtest.py --fast --tag fast

backtest:
	$(PY) scripts/run_backtest.py --tag full

backtest-tuned:
	$(PY) scripts/run_backtest.py --tuned --tag full-tuned

tune-%:
	$(PY) scripts/tune.py --model $*

forecast:
	$(PY) scripts/forecast.py
