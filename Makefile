test:
	poetry run pytest

test.download:
	CI=INTEGRATION poetry run pytest -k test_load_func

cov:
	poetry run pytest --cov-report html
	open htmlcov/index.html

format:
	poetry run pre-commit run --all-files

html:
	open http://localhost:8000/
	poetry run mkdocs serve
