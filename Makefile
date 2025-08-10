.PHONY: demo venv install clean
PY?=python3

venv:
	$(PY) -m venv .venv

install: venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

demo: install
	. .venv/bin/activate && python -m surgicalai_demo --samples_dir data/samples --out runs/demo

clean:
	rm -rf .venv runs/demo feats *.duckdb
