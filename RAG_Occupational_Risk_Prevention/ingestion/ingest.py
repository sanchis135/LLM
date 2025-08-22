# Execution:

# PDFs with text layer or automatic OCR
#python -m ingestion.ingest --raw_dir docs --kb_out data/kb/bm25.jsonl

# Force OCR (if you know they are scanned)
#python -m ingestion.ingest --raw_dir docs --kb_out data/kb/bm25.jsonl --force_ocr
#python -m ingestion.index_vectors --kb_jsonl data/kb/bm25.jsonl --persist_dir data/chroma --collection osha

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Iterable, Dict, Optional

# ---------- Cleaning and tokenization ----------

def clean_text(s: str) -> str:
    # Collapse whitespace and normalize line breaks
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE)
    return s.strip()

def tokenize_words(s: str) -> List[str]:
    # Robust word tokenization (lowercase, multilingual \w)
    return re.findall(r"\w+", s.lower(), flags=re.UNICODE)

# ---------- Metadata heuristics ----------

def infer_family(filename: str) -> str:
    fn = filename.lower()
    if fn.startswith(("law_", "law ")): return "LAW"
    if fn.startswith(("rd_", "royal_decree", "royal-decree", "rd ")): return "ROYAL_DECREE"
    if "osha" in fn or "guideline" in fn: return "GUIDELINE"
    if fn.endswith(".pdf"): return "PDF"
    if fn.endswith(".html") or fn.endswith(".htm"): return "HTML"
    return "TXT"

def infer_year(filename: str) -> Optional[int]:
    # Look for a year like 1995..2099 in the filename
    m = re.search(r"\b(19[5-9]\d|20\d{2})\b", filename)
    try:
        return int(m.group(1)) if m else None
    except Exception:
        return None

# ---------- File parsing (text/HTML/PDF with OCR fallback) ----------

def parse_pdf_textlayer(path: Path) -> str:
    """Extract text from PDFs with text layer (pypdf). If it fails, return ''. """
    try:
        from pypdf import PdfReader
        txt_parts: List[str] = []
        reader = PdfReader(str(path))
        for page in reader.pages:
            txt_parts.append(page.extract_text() or "")
        return clean_text(" ".join(txt_parts))
    except Exception:
        return ""

def parse_pdf_ocr(path: Path, dpi: int = 200, lang: str = "eng") -> str:
    """OCR page by page (pdf2image + pytesseract)."""
    try:
        import pytesseract
        tcmd = os.getenv("TESSERACT_CMD")
        if tcmd:
            pytesseract.pytesseract.tesseract_cmd = tcmd
        from pdf2image import convert_from_path
        images = convert_from_path(str(path), dpi=dpi)
        out = []
        for img in images:
            out.append(pytesseract.image_to_string(img, lang=lang) or "")
        return clean_text(" ".join(out))
    except Exception as e:
        raise RuntimeError(f"OCR failed for {path.name}: {e}")

def parse_pdf(path: Path, force_ocr: bool = False) -> str:
    if force_ocr:
        return parse_pdf_ocr(path)
    txt = parse_pdf_textlayer(path)
    if txt.strip():
        return txt
    # If no text, fallback to OCR
    return parse_pdf_ocr(path)

def parse_html(path: Path) -> str:
    from bs4 import BeautifulSoup
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return clean_text(text)

def parse_txt(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return clean_text(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))

def parse_file(path: Path, force_ocr: bool = False) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return parse_pdf(path, force_ocr=force_ocr)
    if suf in {".html", ".htm"}:
        return parse_html(path)
    if suf == ".txt":
        return parse_txt(path)
    raise ValueError(f"Unsupported extension: {suf} ({path})")

# ---------- Chunking by tokens ----------

def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)

def chunk_text_tokens(text: str, max_tokens: int = 220, overlap: int = 40) -> List[str]:
    """
    Split by tokens (words), with overlap.
    max_tokens: target chunk size
    overlap: tokens shared between consecutive chunks
    """
    assert max_tokens > 0 and overlap >= 0 and overlap < max_tokens
    toks = tokenize_words(text)
    if not toks:
        return []
    chunks: List[str] = []
    i = 0
    step = max_tokens - overlap
    while i < len(toks):
        window = toks[i : i + max_tokens]
        chunks.append(detokenize(window))
        i += step
    return chunks

# ---------- Build KB (BM25 JSONL) ----------

def iter_raw_files(raw_dir: Path) -> Iterable[Path]:
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".html", ".htm", ".txt"}:
            yield p

def build_kb(
    raw_dir: str | Path = "data/raw",
    kb_out: str | Path = "data/kb/bm25.jsonl",
    max_tokens: int = 220,
    overlap: int = 40,
    force_ocr: bool = False,
    verbose: bool = True,
) -> int:
    """
    Read files from raw_dir, create chunks and write JSONL for BM25:
    {"text": str, "tokens": List[str], "source": str, "meta": dict}
    Returns number of chunks written.
    """
    raw_dir = Path(raw_dir)
    kb_out = Path(kb_out)
    kb_out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with kb_out.open("w", encoding="utf-8") as f:
        for path in iter_raw_files(raw_dir):
            try:
                text = parse_file(path, force_ocr=force_ocr)
                if verbose:
                    print(f"[INFO] {path.name}: {len(text)} chars")
                chunks = chunk_text_tokens(text, max_tokens=max_tokens, overlap=overlap)
                if verbose:
                    print(f"[INFO] {path.name}: {len(chunks)} chunks")
                fam = infer_family(path.name)
                yr = infer_year(path.name)
                for j, ch in enumerate(chunks):
                    obj = {
                        "text": ch,
                        "tokens": tokenize_words(ch),
                        "source": f"{path.name}#chunk{j}",
                        "meta": {"file": path.name, "family": fam, "year": yr},
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as e:
                print(f"[WARN] {path}: {e}")
    return count

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Ingestion pipeline: parsing + chunking + BM25 KB (JSONL) with PDF OCR")
    ap.add_argument("--raw_dir", default="data/raw", help="Directory with PDFs/HTML/TXT")
    ap.add_argument("--kb_out", default="data/kb/bm25.jsonl", help="Output JSONL for BM25")
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--force_ocr", action="store_true", help="Force OCR for PDFs (ignore text layer)")
    ap.add_argument("--quiet", action="store_true", help="Less logging")
    args = ap.parse_args()

    n = build_kb(
        raw_dir=args.raw_dir,
        kb_out=args.kb_out,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        force_ocr=args.force_ocr,
        verbose=not args.quiet,
    )
    print(f"✅ KB created with {n} chunks → {args.kb_out}")

if __name__ == "__main__":
    main()
