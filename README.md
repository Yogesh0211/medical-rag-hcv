# HCV RAG – Organized Project
Tools & Skills: Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Python, PyTorch, BM25F, Quantization, Clinical NLP, Evaluation Metrics
Course: Artificial Neural Computation – Arizona State University

Designed and implemented MedRAG, a two-layer RAG framework for medical question answering using social media data (Reddit), enabling low-resource deployment on quantized LLMs.
Achieved performance parity with GPT-4 across coverage, coherence, relevance, and hallucination metrics while operating on consumer-grade hardware with an 8-bit quantized model.
Conducted expert-driven evaluation, statistical analysis, and readability assessment, demonstrating the system’s viability for resource-constrained clinical and patient-facing applications.

## Folder Structure

```
hcv_rag/
├── notebooks/
│   └── HCV_RAG_End_to_End.ipynb         # Clean, step-by-step Colab notebook
├── src/
│   ├── __init__.py
│   ├── config.py                         # Central paths & constants
│   ├── preprocess.py                     # PDF -> text -> cleaned -> chunks.csv
│   ├── retrieval.py                      # Embeddings + FAISS index + search
│   ├── generation.py                     # FLAN-T5 generator + RAG wrapper
│   └── evaluation.py                     # ROUGE/BLEU + grounding checks
├── app/
│   └── gradio_app.py                     # (optional) simple UI
├── data/
│   └── guidelines/                       # Put your PDF guidelines here
├── artifacts/                            # Saved FAISS index, embeddings, chunks.csv
└── outputs/                              # Evaluation reports, logs
```

## Quick Start (Colab)

1. Upload this folder to **Google Drive** and rename the top-level to **`hcv_rag`** (no spaces, no `(1)`).
2. Open **`notebooks/HCV_RAG_End_to_End.ipynb`** in Colab.
3. Run cells top‑to‑bottom. Set your base folder to `/content/drive/MyDrive/hcv_rag` to keep names consistent.
4. Put your PDFs (EASL 2020 + AASLD-IDSA 2023) in `data/guidelines/` (inside your Drive folder).

> If you already have a folder like `rag_project (1)`, you can **rename** it in the Colab notebook using `os.rename()`
> after mounting Drive, or manually in Drive UI **before** running the pipeline.
