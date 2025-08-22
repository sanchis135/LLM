# RAG-PRL: Retrieval-Augmented Generation for Occupational Risk Prevention  

This project implements a **Retrieval-Augmented Generation (RAG)** system focused on **Occupational Risk Prevention (ORP)**.  
It combines **BM25 lexical search**, **vector embeddings** (Sentence Transformers + ChromaDB), and a **hybrid retriever** (Reciprocal Rank Fusion).  
A **Streamlit interface** allows interactive queries with contextualized LLM answers.   

---

## 📂 Repository Structure  

```
rag-prl/
│
├── app/
│   └── ui_streamlit/         # Streamlit interface (Home.py + Dashboard)
│
├── ingestion/                # Scripts for ingestion & indexing
│
├── retrieval/                # BM25, Vector, Hybrid retrievers
│
├── evaluation/               # Retrieval evaluation scripts + metrics
│
├── monitoring/               # Logging & feedback storage (SQLite)
│
├── data/
│   ├── raw/                  # Input docs 
│   ├── kb/                   # BM25 index (.jsonl)
│   ├── chroma/               # Chroma vector DB
│   └── monitoring/           # telemetry.db with logs
│
├── reports/                  # Evaluation results
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation  

```bash
git clone https://github.com/sanchis135/rag-orp.git
cd RAG_Occupational_Risk_Prevention
python -m venv .venv
& .venv\Scripts\activate   # on Windows
pip install -r requirements.txt
```

---

## 📑 Preparing Knowledge Base  

1. Put your documents inside `data/raw`.  
   Example: `data/raw/OSHA2001.pdf`  

2. Run ingestion (BM25 index + vector DB):  

```bash
#Ingest
python -m ingestion.ingest --raw_dir data/raw --kb_out data/kb/bm25.jsonl
#Vectorial Index
python -m ingestion.index_vectors --kb_jsonl data/kb/bm25.jsonl --persist_dir data/chroma --collection osha
#Testing
python scripts/show_chunks.py --doc OSHA2001.pdf --chars 300
```

---

## 🧪 Retrieval Evaluation  

You can evaluate retrieval with example queries:

1. Create a small gold dataset (evaluation/datasets/osha_gold.jsonl) of 8 queries mapped to authoritative passages from the OSH Act:

```bash
python -m evaluation.make_goldset_from_kb --kb_jsonl data/kb/bm25.jsonl --queries_keywords evaluation/datasets/osha_queries_keywords.json --out_jsonl evaluation/datasets/osha_gold.jsonl
```

2. Retrieval was evaluated using Recall@k, comparing BM25, dense retrieval (Chroma + e5 embeddings), and a hybrid approach (RRF):

```bash
python -m evaluation.eval_retrieval   --queries evaluation/datasets/osha_gold.jsonl   --kb_bm25 data/kb/bm25.jsonl   --chroma_dir data/chroma   --collection osha   --k_list 5,10   --out_csv reports/retrieval_eval.csv
```

Results:

=== Mean Recall ===
    recall_bm25  recall_vec  recall_hyb
k
5         0.086       0.143       0.114
10        0.143       0.286       0.200


I constructed a keyword/regex-based gold set over the KB (independent from the retrievers) and measured mean Recall@k on 8 OSH Act questions. Results show dense retrieval outperforming lexical BM25, with hybrid (RRF) in between:

- Recall@5 — BM25: 0.086, Vector: 0.143, Hybrid: 0.114
- Recall@10 — BM25: 0.143, Vector: 0.286, Hybrid: 0.200

Takeaway: For this corpus and chunking, dense retrieval provides higher recall; hybrid fusion helps over BM25 but does not yet surpass dense at k=5/10.

---
## 🧠 LLM Evaluation

Two different prompt styles were tested to generate answers:

1. **Strict style (default)**
- Concise answer in English
- Answers only with cited passages
- Responds “Not found in the indexed sources” if evidence is missing
2. **Structured style**
- Outputs: Summary (2-3 lines), Key Points (bullets), Sources
- Enforces citations [n] for each claim

Switching between prompts is controlled via environment variable:

```bash
# Strict prompt
$env:PROMPT_STYLE="strict"; streamlit run app/ui_streamlit/Home.py

# Structured prompt
$env:PROMPT_STYLE="structured"; streamlit run app/ui_streamlit/Home.py
```

👉 In practice, the structured style produced clearer answers for OSHA regulations, while the strict style was safer for factual queries.

---

## 🚀 Running the Streamlit App  

```bash
streamlit run app/ui_streamlit/Home.py
```

Features:  
- 🔎 **Retrieval tab** → BM25 / Vector / Hybrid search  
- 🧠 **Answer tab** → LLM answers grounded on retrieved passages  
- 📊 **Dashboard** → interaction logs + feedback  

---

## 📝 Logging & Monitoring  

All queries, answers and feedback are stored in:  

```bash
data/monitoring/telemetry.db
```  
You can explore interactions via the **Dashboard tab** in Streamlit.  

The **Metrics tab** shows latency, recall@k, user feedback, etc. with 5+ visualizations.

---

## 🤖 LLM Backend  

The project can run with:  
- **OpenAI GPT models** → requires `OPENAI_API_KEY`  
- **Ollama (local models)** → requires `ollama serve` running  

By default, the code uses **Ollama (llama3.1)**.  

Example test:  

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Say hello in English"
}'
```

---

## 🐳 Containerization

The project is fully containerized with **Docker** and **docker-compose**.

### Build & Run

```bash
docker compose build
docker compose up
```

- App runs on http://localhost:8501
- Data is mounted to ./data and ./reports
- Default model: llama3.1 (Ollama)

### Example ingestion inside the container

```bash
docker compose exec rag_app bash -lc "python -m ingestion.ingest --raw_dir data/raw --kb_out data/kb/bm25.jsonl && python -m ingestion.index_vectors --kb_jsonl data/kb/bm25.jsonl --persist_dir data/chroma --collection osha"
```

## 📌 Example Queries  

- *"What is the purpose of the OSH Act?" (Section 2: Findings and Purpose)*  
- *"Who is responsible for ensuring worker safety?" (Section 5 – Duties)*  
- *"What are workers’ rights under the OSH Act?" (workers’ rights overview)*
- *"What is the General Duty Clause?" (Section 5(a)(1))*  

Run batch QA:

```bash
python -m scripts.batch_qa --questions evaluation/datasets/ad_hoc_questions.jsonl --kb_bm25 data/kb/bm25.jsonl --chroma_dir data/chroma --collection prl --topk 6 --fanout 60 --max_ctx 6 --out_csv reports/batch_qa.csv --out_jsonl reports/batch_qa.jsonl
```

## ✅ Current Status

- Problem: clearly defined (Occupational Risk Prevention + OSHA/PRL regulations)
- Retrieval flow: BM25, Vector, Hybrid (RRF)
- Evaluation: retrieval recall@k computed (dataset still limited)
- LLM evaluation: tested two prompt styles
- Interface: full Streamlit app + dashboard + metrics
- Ingestion: automated via scripts
- Monitoring: feedback + 5+ charts
- Containerization: Dockerfile + docker-compose
- Reproducibility: instructions in README, data accessible via data/raw