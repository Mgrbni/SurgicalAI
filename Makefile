.PHONY: demo install run

demo:
@test -f .env || (echo "Copy .env.example to .env and fill OPENAI_API_KEY"; exit 1)
python -m venv .venv && . .venv/bin/activate && pip install -U pip
. .venv/bin/activate && pip install -r requirements.txt
. .venv/bin/activate && uvicorn server.server:app --host 0.0.0.0 --port 8000

install:
python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
@test -f .env || (echo "Copy .env.example to .env and fill OPENAI_API_KEY"; exit 1)
. .venv/bin/activate && uvicorn server.server:app --host 0.0.0.0 --port 8000

