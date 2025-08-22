# Running:
# python -m evaluation.make_goldset_from_kb --kb_jsonl data/kb/bm25.jsonl --queries_keywords evaluation/datasets/osha_queries_keywords.json --out_jsonl evaluation/datasets/osha_gold.jsonl

from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
from typing import List, Dict

def normalize_text(s: str) -> str:
    s = s.lower()
    # quita diacríticos
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    # colapsa espacios
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def load_kb(kb_jsonl: Path) -> List[Dict]:
    rows = []
    with kb_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            txt = normalize_text(obj.get("text", ""))
            obj["_norm"] = txt
            rows.append(obj)  # {text, tokens, source, meta, _norm}
    return rows

def contains_all(text: str, terms: List[str]) -> bool:
    return all(t.lower() in text for t in terms)

def contains_any(text: str, terms: List[str]) -> bool:
    return any(t.lower() in text for t in terms)

def match_any_regex(text: str, patterns: List[str]) -> bool:
    for pat in patterns:
        try:
            if re.search(pat, text, flags=re.IGNORECASE):
                return True
        except re.error:
            pass
    return False

def main():
    ap = argparse.ArgumentParser(description="Build gold queries from KB using keyword/regex matching.")
    ap.add_argument("--kb_jsonl", default="data/kb/bm25.jsonl")
    ap.add_argument("--queries_keywords", default="evaluation/datasets/osha_queries_keywords.json")
    ap.add_argument("--out_jsonl", default="evaluation/datasets/osha_gold.jsonl")
    ap.add_argument("--max_per_query", type=int, default=5, help="Max gold chunks per query")
    args = ap.parse_args()

    kb = load_kb(Path(args.kb_jsonl))
    spec = json.loads(Path(args.queries_keywords).read_text(encoding="utf-8"))

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as fo:
        for item in spec.get("queries", []):
            q = item["query"]
            all_terms = item.get("keywords_all", [])
            any_terms = item.get("keywords_any", [])
            any_regex = item.get("keywords_any_regex", [])

            gold = []
            for row in kb:
                txt = row["_norm"]

                ok_all = contains_all(txt, [normalize_text(t) for t in all_terms]) if all_terms else True
                ok_any = contains_any(txt, [normalize_text(t) for t in any_terms]) if any_terms else False
                ok_rx  = match_any_regex(txt, any_regex) if any_regex else False

                if ok_all and (ok_any or ok_rx or (not any_terms and not any_regex)):
                    gold.append(row.get("source", "?"))
                    if len(gold) >= args.max_per_query:
                        break

            fo.write(json.dumps({"query": q, "gold_sources": gold}, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"✅ Wrote {n_written} queries → {out_path}")

if __name__ == "__main__":
    main()
