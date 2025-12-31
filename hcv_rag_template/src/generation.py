from typing import Tuple, List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd

RAG_PROMPT = """You are a clinical question-answering assistant.
Answer ONLY using the information in the CONTEXT. If the answer is not in the context, say:
"Not enough evidence in the provided guidelines."

Rules:
- Be concise and factual.
- Prefer bullet points where helpful.
- Include citations like [source: <file>, chunk: <id>] after the statements they support.
- Do NOT invent facts, medications, or durations not present in the context.
- If multiple options exist, list them with the conditions that apply.

QUESTION:
{question}

CONTEXT:
{context}

Now answer grounded ONLY in the context above.
"""

def build_context(rows: pd.DataFrame, max_chars: int = 3500) -> str:
    parts, total = [], 0
    for _, r in rows.iterrows():
        block = f"[rank {r['rank']}] [source: {r['source']} | chunk: {r['chunk_id']}]\n{r['text']}\n"
        if total + len(block) <= max_chars:
            parts.append(block)
            total += len(block)
        else:
            break
    return "\n".join(parts)

def load_generator(model_name: str = None):
    device = 0 if torch.cuda.is_available() else -1
    model_name = model_name or ("google/flan-t5-large" if device == 0 else "google/flan-t5-base")
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == 0 else torch.float32
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=device)
    return gen, model_name

def rag_answer(question: str, retrieved_rows: pd.DataFrame, generator, max_new_tokens: int = 256, temperature: float = 0.0) -> Tuple[str, list, pd.DataFrame]:
    ctx = build_context(retrieved_rows, max_chars=3500)
    prompt = RAG_PROMPT.format(question=question, context=ctx)
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        num_beams=4 if temperature == 0.0 else 1
    )[0]["generated_text"]
    # Return answer, evidence list (strings), and the refs table
    evidence_blocks = [f"{r['source']} | chunk {r['chunk_id']}: {r['text']}" for _, r in retrieved_rows.iterrows()]
    return out, evidence_blocks, retrieved_rows
