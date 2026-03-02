.PHONY: ruff release-pypi

ruff:
	uvx ruff check --fix
	uvx ruff format

release-pypi:
	rm -rf dist
	uvx --with build --with twine python -m build
	uvx --with twine python -m twine upload dist/* $(PYPI_TOKEN)