"""
Tone scoring for earnings call transcripts.

Two-layer approach:
  1. Loughran-McDonald (LM) financial word list — academic standard for
     financial text sentiment. Free, fast, covers all transcripts.
     Source: https://sraf.nd.edu/loughranmcdonald-master-dictionary/

  2. GPT-4o-mini — nuanced features on a sample: hedging intensity,
     evasiveness in Q&A, forward guidance vagueness. ~$5 total.

Output columns added to transcripts DataFrame:
  lm_negative_pct, lm_positive_pct, lm_uncertain_pct, lm_litigious_pct,
  lm_constraining_pct, lm_tone (= positive - negative, normalized),
  gpt_hedging_score, gpt_evasiveness_score, gpt_guidance_vagueness_score
  [gpt columns only on sampled rows]
"""

import re
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from src.config import PROCESSED_DIR, LLM_MODEL, OPENAI_API_KEY

LM_DICT_PATH = PROCESSED_DIR / "lm_dictionary.parquet"

# SRAF moved the dictionary to Google Drive (as of 2024)
# Source: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
LM_DICT_GDRIVE_ID = "1cfg_w3USlRFS97wo7XQmYnuzhpmzboAY"


# ── LM Word List ──────────────────────────────────────────────────────────────

def load_lm_dictionary() -> dict[str, dict]:
    """
    Load Loughran-McDonald master dictionary.
    Downloads and caches as parquet on first run.
    Returns dict: word -> {negative, positive, uncertainty, litigious, constraining}
    """
    if LM_DICT_PATH.exists():
        df = pd.read_parquet(LM_DICT_PATH)
    else:
        print("Downloading LM Master Dictionary...")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Download from Google Drive (SRAF moved the file there in 2024)
        import io
        session = requests.Session()
        url = f"https://drive.google.com/uc?export=download&id={LM_DICT_GDRIVE_ID}"
        resp = session.get(url, timeout=60)
        # handle Google Drive virus-scan confirmation for large files
        token = next(
            (v for k, v in resp.cookies.items() if k.startswith("download_warning")),
            None,
        )
        if token:
            resp = session.get(f"{url}&confirm={token}", timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_parquet(LM_DICT_PATH, index=False)
        print(f"Saved LM dictionary ({len(df):,} words)")

    # Columns: Word, Negative, Positive, Uncertainty, Litigious, Constraining, ...
    # Non-zero value in a category = word belongs to that category
    word_dict = {}
    for _, row in df.iterrows():
        if not isinstance(row["Word"], str):
            continue
        word_dict[row["Word"].upper()] = {
            "negative": int(row["Negative"]) > 0,
            "positive": int(row["Positive"]) > 0,
            "uncertain": int(row["Uncertainty"]) > 0,
            "litigious": int(row["Litigious"]) > 0,
            "constraining": int(row["Constraining"]) > 0,
        }
    return word_dict


def score_text_lm(text: str, lm_dict: dict) -> dict:
    """
    Score a single text string using the LM word list.
    Returns normalized counts (pct of total words) for each category.
    """
    if not text or not isinstance(text, str):
        return {
            "lm_negative_pct": np.nan,
            "lm_positive_pct": np.nan,
            "lm_uncertain_pct": np.nan,
            "lm_litigious_pct": np.nan,
            "lm_constraining_pct": np.nan,
            "lm_tone": np.nan,
            "lm_word_count": 0,
        }

    words = re.findall(r"[A-Za-z]+", text.upper())
    n = len(words)
    if n == 0:
        return {k: np.nan for k in [
            "lm_negative_pct", "lm_positive_pct", "lm_uncertain_pct",
            "lm_litigious_pct", "lm_constraining_pct", "lm_tone"
        ]} | {"lm_word_count": 0}

    counts = {"negative": 0, "positive": 0, "uncertain": 0,
              "litigious": 0, "constraining": 0}
    for w in words:
        if w in lm_dict:
            for cat, flag in lm_dict[w].items():
                if flag:
                    counts[cat] += 1

    return {
        "lm_negative_pct": counts["negative"] / n * 100,
        "lm_positive_pct": counts["positive"] / n * 100,
        "lm_uncertain_pct": counts["uncertain"] / n * 100,
        "lm_litigious_pct": counts["litigious"] / n * 100,
        "lm_constraining_pct": counts["constraining"] / n * 100,
        # tone: positive - negative, normalized to [-100, 100]
        "lm_tone": (counts["positive"] - counts["negative"]) / n * 100,
        "lm_word_count": n,
    }


def score_all_transcripts_lm(df: pd.DataFrame) -> pd.DataFrame:
    """Apply LM scoring to all transcripts. Scores full text, prepared, and Q&A separately."""
    lm_dict = load_lm_dictionary()
    print(f"Scoring {len(df):,} transcripts with LM word list...")

    for col_prefix, text_col in [
        ("full", "full_text"),
        ("prepared", "prepared_text"),
        ("qa", "qa_text"),
    ]:
        scores = df[text_col].apply(lambda t: score_text_lm(t, lm_dict))
        scores_df = pd.DataFrame(scores.tolist())
        scores_df.columns = [f"{col_prefix}_{c}" for c in scores_df.columns]
        df = pd.concat([df, scores_df], axis=1)

    return df


# ── GPT-4o-mini nuanced scoring ───────────────────────────────────────────────

GPT_SYSTEM_PROMPT = """You are a financial analyst scoring earnings call transcripts.
Score the following earnings call excerpt on three dimensions, each from 0 to 10:

1. hedging_score: How much does management hedge or qualify statements?
   (0 = very direct, 10 = extremely hedged / full of qualifiers)

2. evasiveness_score: How evasive are management's answers to analyst questions?
   (0 = very direct answers, 10 = completely avoids the question)

3. guidance_vagueness_score: How vague is forward-looking guidance?
   (0 = specific numbers/dates given, 10 = no concrete guidance at all)

Return ONLY a JSON object with these three keys and numeric values. No explanation."""


def score_transcript_gpt(text: str, client) -> dict:
    """Score a single transcript excerpt with GPT-4o-mini."""
    # Truncate to ~3000 words to control cost
    words = text.split()
    excerpt = " ".join(words[:3000])

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": excerpt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "gpt_hedging_score": result.get("hedging_score"),
            "gpt_evasiveness_score": result.get("evasiveness_score"),
            "gpt_guidance_vagueness_score": result.get("guidance_vagueness_score"),
        }
    except Exception as e:
        print(f"  GPT error: {e}")
        return {
            "gpt_hedging_score": np.nan,
            "gpt_evasiveness_score": np.nan,
            "gpt_guidance_vagueness_score": np.nan,
        }


def score_sample_gpt(df: pd.DataFrame, n_sample: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Score a random sample of transcripts with GPT-4o-mini for nuanced features.
    Adds gpt_* columns (NaN for unsampled rows).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=OPENAI_API_KEY)

    df["gpt_hedging_score"] = np.nan
    df["gpt_evasiveness_score"] = np.nan
    df["gpt_guidance_vagueness_score"] = np.nan

    sample_idx = df.sample(n=min(n_sample, len(df)), random_state=seed).index
    print(f"Scoring {len(sample_idx)} transcripts with GPT-4o-mini...")

    for idx in tqdm(sample_idx, desc="GPT scoring"):
        text = df.loc[idx, "full_text"]
        scores = score_transcript_gpt(text, client)
        for col, val in scores.items():
            df.loc[idx, col] = val
        time.sleep(0.1)  # rate limit buffer

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_tone_scores(force: bool = False, run_gpt: bool = True) -> pd.DataFrame:
    """
    Full tone scoring pipeline.
    Loads transcripts.parquet → scores → saves tone_scores.parquet.
    """
    out_path = PROCESSED_DIR / "tone_scores.parquet"

    if out_path.exists() and not force:
        print(f"Loading cached tone scores from {out_path}")
        return pd.read_parquet(out_path)

    transcripts_path = PROCESSED_DIR / "transcripts.parquet"
    if not transcripts_path.exists():
        raise FileNotFoundError(
            "transcripts.parquet not found — run transcript_loader.py first"
        )

    df = pd.read_parquet(transcripts_path)
    df = score_all_transcripts_lm(df)

    if run_gpt:
        df = score_sample_gpt(df)

    df.to_parquet(out_path, index=False)
    print(f"Saved tone scores for {len(df):,} transcripts to {out_path}")
    return df


if __name__ == "__main__":
    df = build_tone_scores(run_gpt=False)  # set run_gpt=True when ready to spend $5
    print(df[[
        "company_name", "call_date",
        "full_lm_negative_pct", "full_lm_uncertain_pct", "full_lm_tone"
    ]].head(10))
    print(f"\nMean tone by year:\n{df.groupby('fiscal_year')['full_lm_tone'].mean()}")
