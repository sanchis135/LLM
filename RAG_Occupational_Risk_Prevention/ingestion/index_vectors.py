from __future__ import annotations
import argparse
from retrieval.vector_client import VectorClient

def main():
    ap = argparse.ArgumentParser(description="Index JSONL KB chunks into Chroma.")
    ap.add_argument("--kb_jsonl", default="data/kb/bm25.jsonl")
    ap.add_argument("--persist_dir", default="data/chroma")
    ap.add_argument("--collection", default="osha")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--reset", action="store_true", help="Clear the collection before indexing")
    args = ap.parse_args()

    vc = VectorClient(persist_dir=args.persist_dir, collection=args.collection)
    if args.reset:
        vc.reset_collection()

    n = vc.index_from_jsonl(args.kb_jsonl, batch_size=args.batch_size)
    print(f"✅ Indexed {n} chunks into Chroma → {args.persist_dir} (collection='{args.collection}')")

if __name__ == "__main__":
    main()

