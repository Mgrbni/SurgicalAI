.PHONY: demo demo-llm test api ui clean

demo:
pip install -e .
surgicalai demo --out outputs/demo

demo-llm:
pip install -e .
surgicalai demo --out outputs/demo --with-llm

test:
pytest -q

api:
python -m surgicalai.api

ui:
python -m surgicalai.ui

clean:
rm -rf outputs
