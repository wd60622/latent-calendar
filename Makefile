test: 
	poetry run pytest --mpl --mpl-baseline-path=tests/baseline

test.download: 
	CI=INTEGRATION poetry run pytest -k test_load_func 

cov: 
	poetry run pytest --mpl --mpl-baseline-path=tests/baseline --cov-report html --cov=latent_calendar tests && open htmlcov/index.html 

format: 
	poetry run pre-commit run --all-files

html: 
	open http://localhost:8000/
	poetry run mkdocs serve
	
