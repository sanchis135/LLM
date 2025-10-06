import os, json, re, hashlib, yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
from markdownify import markdownify as md
import fitz  # PyMuPDF for PDFs

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
PROC.mkdir(parents=True, exist_ok=True)

@dataclass
class DocChunk:
    doc_id: str
    section: str | None
    text: str
    source: str

def _hash(txt: str) -> str:
    return hashlib.md5(txt.encode('utf-8')).hexdigest()[:10]

def split_markdown(md_text: str) -> List[Dict]:
    # Split by headings; keep section titles
    parts = re.split(r"^#\s+|^##\s+|^###\s+", md_text, flags=re.M)
    chunks = []
    # parts like: ['', 'Title', 'Body', 'SubTitle', 'Body', ...]
    for i in range(1, len(parts), 2):
        section = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ''
        # Windowing to reasonable size
        for win in re.findall(r"(.{1,1200})(?:\n\n|$)", body, flags=re.S):
            chunks.append({'section': section, 'text': win.strip()})
    return chunks

def parse_pdf_to_markdown(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = "\n\n".join([page.get_text("text") for page in doc])
    return md(text)

def main():
    cfg = yaml.safe_load((ROOT / 'ingest' / 'sources.yaml').read_text(encoding='utf-8'))
    all_chunks: List[Dict] = []

    for item in cfg.get('docs', []):
        did = item['id']
        rtype = item['type']
        source = item.get('url') or item.get('path', '')

        if rtype == 'local_markdown':
            md_text = Path(item['path']).read_text(encoding='utf-8')
        elif rtype == 'url_pdf':
            # Optional: download
            import urllib.request
            pdf_out = RAW / f"{did}.pdf"
            pdf_out.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(item['url'], pdf_out)
            except Exception as e:
                print(f"[WARN] Download failed {item['url']}: {e}")
            if not pdf_out.exists():
                print(f"[WARN] Missing {pdf_out}; skip {did}")
                continue
            md_text = parse_pdf_to_markdown(pdf_out)
        else:
            print(f"[WARN] Unknown type {rtype} — skip {did}")
            continue

        for c in split_markdown(md_text):
            all_chunks.append(asdict(DocChunk(
                doc_id=did,
                section=c['section'],
                text=c['text'],
                source=source,
            )))

    out_path = PROC / 'kb_chunks.jsonl'
    with out_path.open('w', encoding='utf-8') as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(all_chunks)} chunks → {out_path}")

if __name__ == '__main__':
    main()
