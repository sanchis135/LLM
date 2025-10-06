"""
monitoring/export_feedback.py

Utility to export user feedback from the local SQLite DB (feedback.sqlite)
to CSV or JSONL for further analysis or model fine-tuning.
"""

import sqlite3
import json
import pandas as pd
import argparse
from pathlib import Path
from collections import Counter
import re

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "monitoring" / "feedback.sqlite"


def load_feedback(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Feedback DB not found at {db_path}")
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM events", con)
    con.close()
    if df.empty:
        raise ValueError("No feedback data found in database.")
    df["payload"] = df["payload"].apply(json.loads)
    flat = pd.json_normalize(df["payload"])
    df = pd.concat([df.drop(columns=["payload"]), flat], axis=1)
    return df


def export_feedback(df: pd.DataFrame, fmt: str = "csv", out_dir: Path | None = None) -> Path:
    out_dir = out_dir or ROOT / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feedback_export.{fmt}"

    if fmt == "csv":
        df.to_csv(out_path, index=False, encoding="utf-8")
    elif fmt == "jsonl":
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError("Format not supported: use 'csv' or 'jsonl'")

    print(f"[OK] Exported {len(df)} feedback rows → {out_path}")
    return out_path


def summarize_feedback(df: pd.DataFrame):
    print("\n=== Feedback Summary ===")
    print(f"🧾 Total records: {len(df)}")
    if "useful" in df.columns:
        print(f"👍 Helpful answers: {sum(df['useful']=='Yes')}")
        print(f"👎 Unhelpful answers: {sum(df['useful']=='No')}")
    if "note" in df.columns and df["note"].notna().any():
        notes = " ".join(df["note"].dropna())
        words = re.findall(r"[a-zA-Z]+", notes.lower())
        top = Counter(words).most_common(5)
        if top:
            print(f"💬 Top feedback words: {', '.join([w for w, _ in top])}")
    print("========================\n")


def main():
    parser = argparse.ArgumentParser(description="Export and summarize feedback from SQLite DB")
    parser.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv", help="Export format")
    parser.add_argument("--outdir", "-o", default=None, help="Output directory")
    args = parser.parse_args()

    df = load_feedback(DB_PATH)
    summarize_feedback(df)
    export_feedback(df, fmt=args.format, out_dir=Path(args.outdir) if args.outdir else None)


if __name__ == "__main__":
    main()
