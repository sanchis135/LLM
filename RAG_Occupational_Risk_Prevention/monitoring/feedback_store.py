
import pathlib, json

FB_PATH = pathlib.Path("data/metrics/feedback.jsonl")
FB_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_feedback(query: str, helpful: bool, comment: str = ""):
    rec = {"query": query, "helpful": helpful, "comment": comment}
    with open(FB_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
