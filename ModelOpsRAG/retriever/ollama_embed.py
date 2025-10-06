import os
from typing import List

try:
    import ollama
except Exception:
    ollama = None

class OllamaEmbeddingFunction:
    def __init__(self, host: str = 'http://127.0.0.1:11434', model: str = 'nomic-embed-text'):
        if ollama is None:
            raise RuntimeError('ollama package not installed')
        os.environ['OLLAMA_HOST'] = host
        self.model = model

    def __call__(self, input: List[str]):
        # Chroma espera un callable que devuelva una lista de vectores
        out = []
        for chunk in input:
            resp = ollama.embeddings(model=self.model, prompt=chunk)
            out.append(resp['embedding'])
        return out
