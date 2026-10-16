.PHONY: install test test-all lint format smoke ingest run run-tfg compare clean

CONFIG ?= configs/us.yaml

install:          ## editable install with dev tools
	pip install -e ".[dev]"

test:             ## fast unit + integration tests (no PPO training)
	pytest -m "not slow"

test-all:         ## everything, with coverage
	pytest --cov --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

format:
	ruff check . --fix && ruff format .

smoke:            ## full pipeline on synthetic data, ~30 s, no network
	regime-rl -v run -c configs/smoke.yaml --source synthetic

ingest:           ## download / top up prices for CONFIG (needs network)
	regime-rl -v ingest -c $(CONFIG)

run:              ## corrected methodology
	regime-rl -v run -c $(CONFIG) --fidelity corrected

run-tfg:          ## reproduce the thesis logic (known defects included)
	regime-rl -v run -c $(CONFIG) --fidelity tfg

compare:          ## both modes side by side from the runs table
	regime-rl query -c $(CONFIG) "SELECT fidelity, created_utc, round(test_alpha,4) AS alpha, round(test_sharpe,2) AS sharpe, round(test_ir,2) AS ir, round(test_max_dd,4) AS max_dd FROM runs ORDER BY created_utc"

clean:
	rm -rf data_smoke outputs_smoke .pytest_cache .ruff_cache .coverage htmlcov
