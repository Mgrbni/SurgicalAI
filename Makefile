.PHONY: demo setup clean sample
VENV?=.venv
PY=$(VENV)/bin/python

setup:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[demo]"

sample:
	$(PY) -m surgicalai.tools.make_sample --out data/lesions_sample --n 8 --seed 1337

demo: setup sample
	$(PY) -m surgicalai.demo --input data/lesions_sample --cpu --offline-llm --out runs/demo
	python -c "import webbrowser, pathlib as p; webbrowser.open(p.Path(runs/demo).resolve().as_uri())"

clean:
	rm -rf $(VENV) runs/demo data/lesions_sample
