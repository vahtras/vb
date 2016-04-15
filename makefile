test:
	python -m pytest --cov=vb --cov-report="html" tests
testv:
	python -m pytest -v --cov=vb --cov-report="html" tests
testx:
	python -m pytest -x --cov=vb --cov-report="html" tests
