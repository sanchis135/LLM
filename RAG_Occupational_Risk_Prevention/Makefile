
.PHONY: up down ingest eval test

up:
	docker compose -f docker/docker-compose.yml up --build

down:
	docker compose -f docker/docker-compose.yml down

ingest:
	python ingestion/flows/parse_clean_index.py --chunk_size 800 --overlap 100

eval:
	python evaluation/eval_retrieval.py

test:
	pytest -q
