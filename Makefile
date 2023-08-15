test: 
	poetry run pytest 

cov: 
	poetry run pytest --cov-report html --cov=latent_calendar tests && open htmlcov/index.html 

format: 
	poetry run black tests latent_calendar

html: 
	open http://localhost:8000/
	poetry run mkdocs serve
	
