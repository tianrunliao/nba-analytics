SRC := src tests

.PHONY: install lint format typecheck test check

install:
	uv sync --frozen

lint:
	uv run --frozen ruff check $(SRC)
	uv run --frozen ruff format --check $(SRC)

format:
	uv run --frozen ruff check --fix $(SRC)
	uv run --frozen ruff format $(SRC)

typecheck:
	uv run --frozen mypy

test:
	uv run --frozen pytest -q

check: lint typecheck test
