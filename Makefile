test: 
	poetry run pytest 

test.download: 
	CI=INTEGRATION poetry run pytest -k test_load_func 

cov: 
	poetry run pytest --cov-report html --cov=latent_calendar tests && open htmlcov/index.html 

format: 
	poetry run black tests latent_calendar scripts

html: 
	open http://localhost:8000/
	poetry run mkdocs serve
	
