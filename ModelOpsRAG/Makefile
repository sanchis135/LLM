.PHONY: setup ingest up eval


setup:
python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt


ingest:
. .venv/bin/activate && python ingest/ingest.py


up:
. .venv/bin/activate && streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0


eval:
. .venv/bin/activate && python eval/evaluate.py