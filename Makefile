.PHONY: install lint format test cov typecheck check clean demo-ieee demo-lending

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ examples/ scripts/
	ruff format --check src/ tests/ examples/ scripts/

format:
	ruff check --fix src/ tests/ examples/ scripts/
	ruff format src/ tests/ examples/ scripts/

typecheck:
	mypy src/

test:
	pytest --cov=drift --cov=integrations --cov-report=term-missing

cov:
	pytest --cov=drift --cov=integrations --cov-report=term-missing --cov-report=html

check: lint typecheck test

demo-ieee:
	python examples/ieee_cis_demo.py

demo-lending:
	python examples/lending_club_demo.py

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .hypothesis .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
