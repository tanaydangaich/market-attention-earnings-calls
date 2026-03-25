"""
Mock data generator for pipeline development and testing.

Generates realistic synthetic data that matches the exact schema
of the real SRAF daily EDGAR data and WRDS transcripts.
When real data arrives, delete mock files and re-run the pipeline — nothing else changes.

Generated files (in data/processed/):
  - edgar_daily_top100.parquet  (matches real SRAF schema)
  - transcripts.parquet         (matches real WRDS schema)
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FIRM_UNIVERSE_PATH

random.seed(42)
np.random.seed(42)

# ── Language pools (realistic earnings call text) ─────────────────────────────

NEGATIVE_PHRASES = [
    "we faced significant headwinds this quarter",
    "revenue declined below our expectations",
    "we are cautious about the near-term outlook",
    "challenging market conditions persisted",
    "we saw deterioration in our core business",
    "margins were impacted by rising costs",
    "we are unable to provide specific guidance at this time",
    "the environment remains difficult and unpredictable",
    "we experienced meaningful weakness in demand",
    "we fell short of our targets this quarter",
    "credit losses increased substantially year over year",
    "we are taking steps to address underperformance",
    "impairment charges weighed on results",
    "liquidity concerns have intensified across the sector",
    "we are evaluating all options to stabilize the business",
]

UNCERTAIN_PHRASES = [
    "it is difficult to predict how conditions will evolve",
    "we remain cautious given the uncertain environment",
    "visibility into the second half is limited",
    "we cannot be certain about the timing of recovery",
    "outcomes are subject to a wide range of variables",
    "we are monitoring the situation closely",
    "the range of potential outcomes remains wide",
    "we may need to revise our assumptions",
    "there are factors beyond our control",
    "we are not in a position to give specific guidance",
]

POSITIVE_PHRASES = [
    "we delivered strong results this quarter",
    "revenue grew ahead of expectations",
    "we are confident in our long-term trajectory",
    "margins expanded driven by operational efficiency",
    "demand trends remain robust across our segments",
    "we generated significant free cash flow",
    "our competitive position has never been stronger",
    "we are executing well against our strategic priorities",
    "customer retention improved meaningfully this quarter",
    "we are raising our full-year guidance",
    "new product launches exceeded our expectations",
    "our balance sheet remains strong and flexible",
]

NEUTRAL_PHRASES = [
    "turning to our segment results",
    "as I mentioned in my prepared remarks",
    "let me provide some additional context",
    "on a year-over-year basis",
    "looking at the balance sheet",
    "capital expenditures were in line with our plan",
    "our effective tax rate was approximately",
    "we repurchased shares during the quarter",
    "depreciation and amortization totaled",
    "I will now turn the call over to questions",
]

QA_ANALYST_QUESTIONS = [
    "Can you provide more color on the margin pressure you mentioned?",
    "What gives you confidence in the second half recovery?",
    "How should we think about capital allocation going forward?",
    "Can you quantify the impact of the headwinds you described?",
    "What is your current liquidity position?",
    "Are you seeing any signs of stabilization in demand?",
    "How do you think about the competitive landscape?",
    "Can you walk us through your guidance assumptions?",
]

QA_EVASIVE_ANSWERS = [
    "We're not going to provide specific numbers at this point, but we're monitoring it closely.",
    "I think it's premature to quantify that. What I would say is we're focused on execution.",
    "We'll have more to share on that at our investor day in the spring.",
    "There are a number of factors at play and we're evaluating the situation holistically.",
]

QA_DIRECT_ANSWERS = [
    "The impact was approximately 50 basis points on gross margin, primarily from input costs.",
    "We expect Q3 to be roughly flat sequentially, with meaningful improvement in Q4.",
    "Our cash position stands at $2.4 billion with no near-term debt maturities.",
    "Guidance assumes 3-5% volume growth and stable pricing in our core markets.",
]


def generate_transcript(
    company_name: str,
    call_date: pd.Timestamp,
    tone: str = "neutral",  # "negative", "neutral", "positive"
    crisis: bool = False,
) -> dict:
    """Generate a single realistic earnings call transcript."""

    if tone == "negative":
        main_phrases = NEGATIVE_PHRASES + UNCERTAIN_PHRASES
        qa_answers = QA_EVASIVE_ANSWERS * 3 + QA_DIRECT_ANSWERS
        n_sentences = random.randint(40, 60)
    elif tone == "positive":
        main_phrases = POSITIVE_PHRASES * 2 + NEUTRAL_PHRASES
        qa_answers = QA_DIRECT_ANSWERS * 3 + QA_EVASIVE_ANSWERS
        n_sentences = random.randint(45, 65)
    else:
        main_phrases = POSITIVE_PHRASES + NEGATIVE_PHRASES + NEUTRAL_PHRASES * 2
        qa_answers = QA_DIRECT_ANSWERS + QA_EVASIVE_ANSWERS
        n_sentences = random.randint(35, 55)

    if crisis:
        main_phrases = NEGATIVE_PHRASES + UNCERTAIN_PHRASES * 2 + main_phrases

    quarter = (call_date.month - 1) // 3 + 1
    year = call_date.year

    prepared = (
        f"Good morning and thank you for joining {company_name}'s "
        f"Q{quarter} {year} earnings call. "
    )
    prepared += " ".join(random.choices(main_phrases, k=n_sentences)) + " "
    prepared += " ".join(random.choices(NEUTRAL_PHRASES, k=10))

    qa_parts = []
    for q in random.sample(QA_ANALYST_QUESTIONS, k=4):
        answer = random.choice(qa_answers)
        qa_parts.append(f"Analyst: {q} Management: {answer}")
    qa_text = " ".join(qa_parts)

    full_text = prepared + " " + qa_text

    return {
        "prepared_text": prepared,
        "qa_text": qa_text,
        "full_text": full_text,
        "word_count": len(full_text.split()),
    }


def generate_transcripts(firms: list[dict]) -> pd.DataFrame:
    """Generate quarterly earnings call transcripts for all firms, 2003-2015."""
    print("Generating mock transcripts...")
    rows = []

    crisis_start = pd.Timestamp("2007-07-01")
    crisis_end = pd.Timestamp("2009-06-30")

    for firm in firms:
        # assign a firm-level "base tone" — some firms are structurally more negative
        firm_base_tone = random.gauss(0, 1)

        for year in range(2003, 2016):
            for quarter in range(1, 5):
                month = quarter * 3
                # earnings calls happen ~3-6 weeks after quarter end
                call_date = pd.Timestamp(year=year, month=month, day=1) + pd.Timedelta(
                    days=random.randint(21, 42)
                )
                if call_date > pd.Timestamp("2015-12-31"):
                    continue

                is_crisis = crisis_start <= call_date <= crisis_end

                # tone influenced by firm base, crisis period, and randomness
                tone_score = firm_base_tone + random.gauss(0, 1.5)
                if is_crisis:
                    tone_score -= 2.0  # crisis → more negative

                if tone_score < -1.0:
                    tone = "negative"
                elif tone_score > 1.0:
                    tone = "positive"
                else:
                    tone = "neutral"

                transcript = generate_transcript(
                    firm["name"], call_date, tone=tone, crisis=is_crisis
                )

                rows.append({
                    "transcript_id": f"{firm['cik']}_{year}_Q{quarter}",
                    "cik": firm["cik"],
                    "ticker": firm["ticker"],
                    "company_name": firm["name"],
                    "call_date": call_date,
                    "fiscal_year": year,
                    "fiscal_quarter": quarter,
                    "is_mock": True,
                    **transcript,
                })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} mock transcripts")
    return df


def generate_edgar_daily(
    firms: list[dict], transcripts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate daily EDGAR download data for all firms, 2003-2015.
    Bakes in the hypothesis: negative tone → download spike in following 30 days.
    """
    print("Generating mock EDGAR daily data...")
    rows = []

    date_range = pd.date_range("2003-01-01", "2015-12-31", freq="D")

    # build a lookup: (cik, date) → tone category from transcripts
    tone_lookup = {}
    for _, t in transcripts_df.iterrows():
        cik = t["cik"]
        call_date = pd.Timestamp(t["call_date"])
        # determine tone from word counts (approximate)
        text = t["full_text"]
        neg_words = sum(1 for p in NEGATIVE_PHRASES for w in p.split() if w in text.lower())
        pos_words = sum(1 for p in POSITIVE_PHRASES for w in p.split() if w in text.lower())
        net_tone = pos_words - neg_words
        tone_lookup[(cik, call_date.date())] = net_tone

    form_types = ["10-K", "10-Q", "10-Q", "10-Q", "8-K", "8-K", "8-K", "DEF 14A"]

    for firm in firms:
        cik = firm["cik"]
        # baseline download rate — proportional to firm's total downloads
        baseline = max(10, firm["total"] // (13 * 365))

        # track upcoming earnings calls for this firm
        firm_calls = transcripts_df[transcripts_df["cik"] == cik][
            ["call_date", "fiscal_year", "fiscal_quarter"]
        ].copy()
        firm_calls["call_date"] = pd.to_datetime(firm_calls["call_date"])

        for date in date_range:
            if date.weekday() >= 5:  # skip weekends
                continue

            # base downloads with noise
            daily_downloads = max(0, int(np.random.poisson(baseline)))

            # check if within 30 days after an earnings call
            for _, call in firm_calls.iterrows():
                days_since_call = (date - call["call_date"]).days
                if 0 <= days_since_call <= 30:
                    tone_net = tone_lookup.get(
                        (cik, call["call_date"].date()), 0
                    )
                    # negative tone → spike; positive tone → smaller effect
                    if tone_net < -5:
                        spike_multiplier = np.random.uniform(1.5, 3.5)
                    elif tone_net < 0:
                        spike_multiplier = np.random.uniform(1.1, 1.8)
                    else:
                        spike_multiplier = np.random.uniform(0.9, 1.2)

                    # decay over 30 days
                    decay = max(0, 1 - days_since_call / 30)
                    daily_downloads = int(
                        daily_downloads * (1 + (spike_multiplier - 1) * decay)
                    )
                    break

            if daily_downloads == 0:
                continue

            # generate 1-3 filing accession records for this firm-day
            n_accessions = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            for _ in range(n_accessions):
                htm = int(daily_downloads * random.uniform(0.5, 0.7))
                txt = int(daily_downloads * random.uniform(0.1, 0.3))
                xbrl = int(daily_downloads * random.uniform(0.05, 0.15))
                other = max(0, daily_downloads - htm - txt - xbrl)

                rows.append({
                    "date": date,
                    "cik": cik,
                    "accession": f"{cik}-{date.strftime('%Y%m%d')}-{random.randint(1000,9999):04d}",
                    "form": random.choice(form_types),
                    "filing_date": date - pd.Timedelta(days=random.randint(1, 90)),
                    "nr_total": daily_downloads,
                    "htm": htm,
                    "txt": txt,
                    "xbrl": xbrl,
                    "other": other,
                    "company_name": firm["name"],
                })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} mock EDGAR daily rows")
    return df


def build_mock_data(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate and save both mock datasets.
    Returns (edgar_df, transcripts_df).
    Skips generation if files already exist unless force=True.
    """
    edgar_path = PROCESSED_DIR / "edgar_daily_top100.parquet"
    transcripts_path = PROCESSED_DIR / "transcripts.parquet"

    if edgar_path.exists() and transcripts_path.exists() and not force:
        print("Mock data already exists. Loading from cache.")
        print("(Pass force=True to regenerate, or delete files to use real data)")
        return (
            pd.read_parquet(edgar_path),
            pd.read_parquet(transcripts_path),
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(FIRM_UNIVERSE_PATH) as f:
        firms = json.load(f)

    transcripts_df = generate_transcripts(firms)
    transcripts_df.to_parquet(transcripts_path, index=False)
    print(f"Saved mock transcripts → {transcripts_path}")

    edgar_df = generate_edgar_daily(firms, transcripts_df)
    edgar_df.to_parquet(edgar_path, index=False)
    print(f"Saved mock EDGAR daily → {edgar_path}")

    return edgar_df, transcripts_df


if __name__ == "__main__":
    edgar_df, transcripts_df = build_mock_data(force=True)
    print(f"\nEDGAR: {len(edgar_df):,} rows, {edgar_df['cik'].nunique()} firms")
    print(f"Transcripts: {len(transcripts_df):,} calls")
    print(f"Date range: {edgar_df['date'].min()} — {edgar_df['date'].max()}")
