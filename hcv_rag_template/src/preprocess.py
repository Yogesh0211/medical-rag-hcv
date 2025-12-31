from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from nltk import sent_tokenize
import pandas as pd
import nltk

def extract_text_from_pdfs(input_dir: Path, output_dir: Path) -> List[Path]:
    """Extract raw text from all PDFs in input_dir into output_dir/*.txt"""
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = []
    for pdf in input_dir.glob("*.pdf"):
        doc = fitz.open(pdf)
        full = []
        for page in doc:
            full.append(page.get_text())
        out = output_dir / f"{pdf.stem}.txt"
        out.write_text("\n".join(full), encoding="utf-8")
        txt_files.append(out)
    return txt_files

def clean_text(text: str) -> str:
    lines = text.split("\n")
    lines = [ln.strip() for ln in lines if ln.strip()]
    return " ".join(lines)

def clean_txt_files(input_dir: Path) -> None:
    """Create *_cleaned.txt for each .txt file in place."""
    for txt in input_dir.glob("*.txt"):
        if txt.name.endswith("_cleaned.txt"):
            continue
        raw = txt.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        out = txt.with_name(txt.stem + "_cleaned.txt")
        out.write_text(cleaned, encoding="utf-8")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks using sentence boundaries."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    sents = sent_tokenize(text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) <= chunk_size:
            cur += " " + s
        else:
            chunks.append(cur.strip())
            cur = " ".join(cur.split()[-overlap:]) + " " + s
    if cur:
        chunks.append(cur.strip())
    return chunks

def make_chunks_csv(cleaned_files: List[Path], out_csv: Path) -> Path:
    rows = []
    for cf in cleaned_files:
        txt = cf.read_text(encoding="utf-8")
        ch = chunk_text(txt, chunk_size=500, overlap=50)
        for i, c in enumerate(ch):
            rows.append({"source": cf.name, "chunk_id": i, "text": c})
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
