import re
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

ABSTAIN_TEXT = "Not enough evidence in the provided guidelines."

def has_citation(text: str) -> bool:
    return bool(re.search(r"\[.*source.*\]", text, flags=re.IGNORECASE))

def extract_numbers(text: str):
    return set(re.findall(r"\b\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?%?\b", text))

def numeric_consistency(answer: str, evidence_blocks: list):
    evidence_text = " ".join(evidence_blocks)
    ans_nums = extract_numbers(answer)
    ev_nums = extract_numbers(evidence_text)
    extra = sorted(list(ans_nums - ev_nums))
    return {
        "numbers_in_answer": sorted(list(ans_nums)),
        "numbers_in_evidence": sorted(list(ev_nums)),
        "numbers_not_in_evidence": extra,
    }

def text_metrics(answer: str, gold: str):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    r = scorer.score(gold, answer)['rougeL']
    bleu = sacrebleu.corpus_bleu([answer], [[gold]]).score
    return {"rougeL_f": r.fmeasure, "bleu": bleu}

def evaluate_one(question: str, gold_ref: str, rag_fn, top_k: int = 6):
    """
    rag_fn: callable that takes (question, top_k) -> (answer, evidence_blocks, refs_df)
    """
    answer, evidence_blocks, refs_df = rag_fn(question, top_k)
    res = {
        "question": question,
        "answer": answer,
        "has_citation": has_citation(answer),
        "abstained": (ABSTAIN_TEXT.lower() in answer.lower())
    }
    if gold_ref:
        tm = text_metrics(answer, gold_ref)
        res.update(tm)
    num = numeric_consistency(answer, evidence_blocks)
    res["numbers_not_in_evidence"] = ", ".join(num["numbers_not_in_evidence"])
    res["evidence_used"] = "\n".join(evidence_blocks)
    return res, refs_df

def evaluate_batch(df_eval: pd.DataFrame, rag_fn, top_k: int = 6):
    rows, all_refs = [], []
    for i, row in df_eval.iterrows():
        q = row["question"]
        gold = row.get("gold", "")
        r, refs = evaluate_one(q, gold, rag_fn, top_k)
        rows.append(r)
        tmp = refs.copy()
        tmp["eval_row"] = i
        all_refs.append(tmp)
    report = pd.DataFrame(rows)
    refs_report = pd.concat(all_refs, ignore_index=True)
    return report, refs_report
