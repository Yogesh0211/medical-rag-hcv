from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_csv: Path) -> pd.DataFrame:
    return pd.read_csv(chunks_csv)

def build_embeddings(texts: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    embed = SentenceTransformer(model_name)
    X = embed.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return X

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index: faiss.IndexFlatL2, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def load_index(path: Path) -> faiss.IndexFlatL2:
    return faiss.read_index(str(path))

def search(index: faiss.IndexFlatL2, df_chunks: pd.DataFrame, query: str, top_k: int = 5, model_name: str="all-MiniLM-L6-v2"):
    embed = SentenceTransformer(model_name)
    q = embed.encode([query], convert_to_numpy=True)
    D, I = index.search(q, top_k)
    rows = df_chunks.iloc[I[0]].copy()
    rows["rank"] = range(1, len(rows)+1)
    return rows, D[0], I[0]
