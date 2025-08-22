# app/llm/generate.py
from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Optional

import requests  # required for Ollama

def _build_prompt_strict(query: str, contexts: list[dict]) -> str:
    lines = []
    lines.append("You are a careful assistant. Answer IN ENGLISH using ONLY the given passages.")
    lines.append('If the answer is not fully supported, say: "Not found in the indexed sources."')
    lines.append("Cite sources with [n] mapping to the passages list.")
    lines.append("")
    lines.append(f"User question: {query}")
    lines.append("")
    lines.append("Context passages:")
    for i, c in enumerate(contexts, start=1):
        lines.append(f"[{i}] {(c.get('text') or '')[:2000]}")
    lines.append("")
    lines.append("Now write a concise, accurate answer in English with citations [n].")
    return "\n".join(lines)

def _build_prompt_structured(query: str, contexts: list[dict]) -> str:
    lines = []
    lines.append("You are an OSHA legal assistant. Answer IN ENGLISH using ONLY the given passages.")
    lines.append("Structure the answer as: Summary (2-3 lines), Key Points (bullets), Sources.")
    lines.append("Cite each claim with [n], refer to the passages below. If missing, say it.")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Passages:")
    for i, c in enumerate(contexts, start=1):
        lines.append(f"[{i}] {(c.get('text') or '')[:2000]}")
    lines.append("")
    lines.append("Write the answer now, structured, concise, with citations [n].")
    return "\n".join(lines)


# =========================
# Prompt helpers
# =========================
def _build_prompt(query: str, contexts: list[dict]) -> str:
    style = (os.getenv("PROMPT_STYLE") or "strict").lower()
    if style == "structured":
        return _build_prompt_structured(query, contexts)
    return _build_prompt_strict(query, contexts)


# =========================
# Ollama (preferred)
# =========================
def _ollama_url() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

def _is_ollama_up(timeout: float = 2.5) -> bool:
    try:
        r = requests.get(f"{_ollama_url()}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def answer_with_ollama(query: str, contexts: List[Dict], model: Optional[str] = None) -> str:
    """
    Generate with Ollama. Requires `ollama serve` running and a model available (e.g. `ollama pull llama3.1`).
    Useful env vars:
      - OLLAMA_MODEL (e.g. 'llama3.1')
      - OLLAMA_HOST  (e.g. 'http://localhost:11434')
      - OLLAMA_TEMPERATURE (optional float)
      - OLLAMA_NUM_CTX (optional int)
    """
    if not _is_ollama_up():
        raise RuntimeError("Ollama not responding at 11434. Is 'ollama serve' running?")

    model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    num_ctx_env = os.getenv("OLLAMA_NUM_CTX")
    options = {"temperature": temperature}
    if num_ctx_env:
        try:
            options["num_ctx"] = int(num_ctx_env)
        except ValueError:
            pass

    prompt = _build_prompt(query, contexts)

    r = requests.post(
        f"{_ollama_url()}/api/generate",
        json={"model": model, "prompt": prompt, "options": options, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


# =========================
# OpenAI (optional fallback)
# =========================
def answer_with_openai(query: str, contexts: List[Dict], model: Optional[str] = None) -> str:
    """
    Uses the new openai>=1.0.0 SDK (optional).
    Env vars:
      - OPENAI_API_KEY
      - OPENAI_MODEL (default: gpt-4o-mini)
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Package 'openai' not available or incompatible. Install 'openai>=1.0.0' if you want to use this fallback.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Configure it or use Ollama instead.")

    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(query, contexts)

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )
    return (resp.choices[0].message.content or "").strip()


# =========================
# Router
# =========================
def generate_answer(query, ctx):
    model = os.getenv("OLLAMA_MODEL", "llama3.1")  # default: llama3.1
    prompt = f"Answer in English, using the following sources:\n\n{ctx}\n\nQuestion: {query}"

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


# Alternative version for OpenAI (commented out)
# from __future__ import annotations
# import os
# from typing import List, Dict
#
# def _build_prompt(query: str, contexts: List[Dict], cite_style: str = "brackets") -> str:
#     lines = []
#     lines.append("You are a careful assistant. Answer the user question IN ENGLISH.")
#     lines.append("Base your answer ONLY on the provided context passages. If key details are missing, say so explicitly and answer only what is supported.")
#     lines.append("Cite sources inline with bracketed numbers like [1], [2] that map to the passages below.")
#     lines.append("")
#     lines.append(f"User question: {query}")
#     lines.append("")
#     lines.append("Context passages:")
#     for i, c in enumerate(contexts, start=1):
#         text = c["text"][:2000]
#         lines.append(f"[{i}] {text}")
#     lines.append("")
#     lines.append("Now write a concise, accurate answer in English with citations [n] next to each claim that uses a passage.")
#     return "\n".join(lines)
#
# # -------- OpenAI (SDK >= 1.0.0) --------
# def answer_with_openai(query: str, contexts: List[Dict], model: str = None) -> str:
#     try:
#         from openai import OpenAI
#     except Exception as e:
#         raise RuntimeError(f"Could not import openai>=1.0.0: {e}")
#
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set")
#
#     model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
#     prompt = _build_prompt(query, contexts)
#
#     try:
#         client = OpenAI(api_key=api_key)
#         resp = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=600,
#         )
#         return (resp.choices[0].message.content or "").strip()
#     except Exception as e:
#         raise RuntimeError(f"OpenAI error: {e}")
#
# # -------- Ollama (local alternative) --------
# def answer_with_ollama(query: str, contexts: List[Dict], model: str = None) -> str:
#     import requests, json
#     model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
#     prompt = _build_prompt(query, contexts)
#
#     r = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": model, "prompt": prompt, "options": {"temperature": 0.2}},
#         timeout=120
#     )
#     r.raise_for_status()
#     text = []
#     for line in r.text.splitlines():
#         try:
#             obj = json.loads(line)
#             if "response" in obj:
#                 text.append(obj["response"])
#         except json.JSONDecodeError:
#             pass
#     return "".join(text).strip()
#
# # -------- Router --------
# def generate_answer(query: str, contexts: List[Dict]) -> str:
#     if os.getenv("OPENAI_API_KEY"):
#         return answer_with_openai(query, contexts)
#     try:
#         return answer_with_ollama(query, contexts)
#     except Exception:
#         return ("Cannot generate the answer automatically because no LLM provider is configured "
#                 "(missing OPENAI_API_KEY and Ollama not responding). "
#                 "However, the retrieved passages are listed above.")
