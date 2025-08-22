
import argparse, pathlib, json, re
from pypdf import PdfReader
from bs4 import BeautifulSoup
from retrieval.utils import tokenize

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_pdf(path: pathlib.Path) -> str:
    text = []
    r = PdfReader(str(path))
    for page in r.pages:
        text.append(page.extract_text() or "")
    return clean_text(" \n".join(text))

def parse_html(path: pathlib.Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script","style","nav","footer","header"]):
        tag.decompose()
    return clean_text(soup.get_text(separator=" \n "))

def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

def main(args):
    raw = pathlib.Path("data/raw")
    kb_bm25 = pathlib.Path("data/kb/bm25.jsonl")
    kb_bm25.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".html", ".htm", ".txt"}:
            try:
                if p.suffix.lower() == ".pdf":
                    txt = parse_pdf(p)
                elif p.suffix.lower() in {".html", ".htm"}:
                    txt = parse_html(p)
                else:
                    txt = clean_text(p.read_text(encoding="utf-8", errors="ignore"))
                chunks = chunk_text(txt, args.chunk_size, args.overlap)
                for j, ch in enumerate(chunks):
                    entries.append({"text": ch, "tokens": tokenize(ch), "source": f"{p.name}#chunk{j}", "meta": {"file": p.name}})
            except Exception as e:
                print(f"[WARN] Processing error {p}: {e}")

    with open(kb_bm25, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # También indexamos en Chroma
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        client = chromadb.Client()
        model = "intfloat/multilingual-e5-small"
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
        col = client.get_or_create_collection("osha", embedding_function=embedder)
        ids = [str(i) for i in range(len(entries))]
        docs = [e["text"] for e in entries]
        metas = [{"source": e["source"], **e.get("meta", {})} for e in entries]
        if len(ids) > 0:
            col.add(ids=ids, documents=docs, metadatas=metas)
        print(f"Indexed {len(entries)} chunks into BM25 and Chroma")
    except Exception as e:
        print(f"[WARN] Failed to index into Chroma: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    main(parser.parse_args())
