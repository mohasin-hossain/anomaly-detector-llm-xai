.PHONY: setup run test fmt clean

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && streamlit run app/Home.py

test:
	. .venv/bin/activate && pytest -q tests/

fmt:
	@echo "Formatting placeholder - add black/ruff if needed"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

