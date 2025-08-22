
import re

def tokenize(text: str):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)
