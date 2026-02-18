.PHONY: ruff

ruff:
	uvx ruff check --fix
	uvx ruff format