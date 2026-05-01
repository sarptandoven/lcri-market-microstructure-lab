.PHONY: install demo test lint clean

install:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip && pip install -e '.[dev]'

demo:
	. .venv/bin/activate && lcri-lab run-demo --rows 20000 --seed 7

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache reports/*.csv reports/*.json reports/*.md reports/figures/*.png
