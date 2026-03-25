"""
Earnings call transcript pipeline via WRDS (Refinitiv StreetEvents).

Pulls earnings call transcripts for the top 100 firms, 2003-2015.
Requires a WRDS account with Refinitiv StreetEvents access.

WRDS table: tr_events.wrds_keydev
Transcript text: tr_events.wrds_transcript_detail (or similar — see notes below)

Output schema (data/processed/transcripts.parquet):
  cik (str), ticker (str), company_name (str),
  call_date (datetime), fiscal_year (int), fiscal_quarter (int),
  transcript_text (str), word_count (int)

Setup:
  1. pip install wrds
  2. Run `python -c "import wrds; wrds.Connection()"` once to store credentials
  3. Or set WRDS_USERNAME in .env and pass password interactively
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
FIRM_UNIVERSE_PATH = ROOT / "data" / "firm_universe.json"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_firm_universe() -> list[dict]:
    with open(FIRM_UNIVERSE_PATH) as f:
        return json.load(f)


def get_wrds_connection():
    """Lazy import so the module loads without wrds installed."""
    try:
        import wrds
        return wrds.Connection()
    except ImportError:
        raise ImportError("Install wrds: pip install wrds")


def fetch_transcripts_for_firm(
    conn, cik: str, ticker: str, start_year: int = 2003, end_year: int = 2015
) -> pd.DataFrame:
    """
    Pull earnings call transcripts for a single firm from WRDS.

    WRDS / Refinitiv StreetEvents tables:
      - ciq_transcripts.ciqtranscript          : transcript metadata
      - ciq_transcripts.ciqtranscriptcomponent  : transcript text by speaker/section

    We join on transcriptid to get full text.
    Earnings calls = keydeveventtypeid = 48 (Earnings Call).
    """
    query = f"""
        SELECT
            t.transcriptid,
            t.companyid,
            t.mostimportantdateutc::date AS call_date,
            t.transcriptcollectiontypename,
            tc.componenttypename,
            tc.speakertypename,
            tc.componenttext
        FROM ciq_transcripts.ciqtranscript t
        JOIN ciq_transcripts.ciqtranscriptcomponent tc
            ON t.transcriptid = tc.transcriptid
        WHERE t.transcriptcollectiontypename = 'Earnings Call'
          AND EXTRACT(YEAR FROM t.mostimportantdateutc) BETWEEN {start_year} AND {end_year}
          AND UPPER(t.companyname) LIKE UPPER('%{ticker.replace("'", "")}%')
        ORDER BY t.mostimportantdateutc, tc.componentorder
    """
    try:
        df = conn.raw_sql(query)
        return df
    except Exception as e:
        print(f"  Warning: failed for {ticker} ({e})")
        return pd.DataFrame()


def aggregate_transcript_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate component-level rows into one row per transcript,
    separating prepared remarks from Q&A.
    """
    if df.empty:
        return df

    results = []
    for tid, group in df.groupby("transcriptid"):
        prepared = group[
            group["componenttypename"].str.contains("Presentation|Prepared", case=False, na=False)
        ]["componenttext"].str.cat(sep=" ")

        qa = group[
            group["componenttypename"].str.contains("Question|Answer|Q&A", case=False, na=False)
        ]["componenttext"].str.cat(sep=" ")

        full_text = group["componenttext"].str.cat(sep=" ")

        results.append({
            "transcript_id": tid,
            "call_date": group["call_date"].iloc[0],
            "prepared_text": prepared,
            "qa_text": qa,
            "full_text": full_text,
            "word_count": len(full_text.split()),
        })

    return pd.DataFrame(results)


def build_transcripts(force: bool = False) -> pd.DataFrame:
    """
    Full pipeline: connect to WRDS → fetch transcripts for all top 100 firms
    → aggregate text → save parquet.
    """
    out_path = PROCESSED_DIR / "transcripts.parquet"

    if out_path.exists() and not force:
        print(f"Loading cached transcripts from {out_path}")
        return pd.read_parquet(out_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    firms = load_firm_universe()

    print("Connecting to WRDS...")
    conn = get_wrds_connection()

    all_transcripts = []
    for firm in tqdm(firms, desc="Fetching transcripts"):
        raw = fetch_transcripts_for_firm(conn, firm["cik"], firm["ticker"])
        if raw.empty:
            continue
        agg = aggregate_transcript_text(raw)
        agg["cik"] = firm["cik"]
        agg["ticker"] = firm["ticker"]
        agg["company_name"] = firm["name"]
        all_transcripts.append(agg)

    if not all_transcripts:
        raise ValueError("No transcripts retrieved — check WRDS access and table names")

    df = pd.concat(all_transcripts, ignore_index=True)
    df["call_date"] = pd.to_datetime(df["call_date"])
    df["fiscal_year"] = df["call_date"].dt.year
    df["fiscal_quarter"] = df["call_date"].dt.quarter

    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} transcripts to {out_path}")
    return df


if __name__ == "__main__":
    df = build_transcripts()
    print(df[["company_name", "call_date", "word_count"]].head(20))
    print(f"\nTotal transcripts: {len(df):,}")
    print(f"Date range: {df['call_date'].min()} — {df['call_date'].max()}")
    print(f"Avg word count: {df['word_count'].mean():.0f}")
