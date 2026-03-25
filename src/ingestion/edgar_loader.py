"""
EDGAR daily download data pipeline.

The Notre Dame SRAF dataset (Loughran & McDonald 2017) provides daily
filing download counts per (cik, accession, date). This module:
  1. Downloads the compressed daily data from SRAF Google Drive
  2. Filters to the top 100 firms by total download volume
  3. Saves a clean parquet file for event study analysis

Raw data schema:
  date, cik, accession, form, filing_date, nr_total, htm, txt, xbrl, other

Output schema (data/processed/edgar_daily_top100.parquet):
  date (datetime), cik (str), accession (str), form (str),
  filing_date (datetime), nr_total (int), htm (int), txt (int),
  xbrl (int), other (int), company_name (str)
"""

import json
import zipfile
import io
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "edgar_daily"
PROCESSED_DIR = ROOT / "data" / "processed"
TOP100_PATH = ROOT / "data" / "top100_firms.json"

# SRAF compressed daily data — hosted on Google Drive
# Source: https://sraf.nd.edu/data/edgar-server-log/
# Direct file ID from the SRAF data page (verify current ID at sraf.nd.edu)
SRAF_GDRIVE_FILE_ID = "1O-jNlkEVdKflgBRSWXMLqagR_HGq5Ck5"
SRAF_FILENAME = "EDGAR_Log_File_Data.zip"


def load_top100_ciks() -> set:
    with open(TOP100_PATH) as f:
        firms = json.load(f)
    return {str(f["cik"]) for f in firms}


def download_sraf_data(dest: Path = RAW_DIR) -> Path:
    """
    Download the SRAF compressed daily EDGAR log data from Google Drive.
    File is ~2.1 GB compressed. Skips download if already present.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / SRAF_FILENAME

    if out_path.exists():
        print(f"Already downloaded: {out_path}")
        return out_path

    print("Downloading SRAF EDGAR daily data (~2.1 GB)...")
    url = f"https://drive.google.com/uc?export=download&id={SRAF_GDRIVE_FILE_ID}"

    # Google Drive requires a confirmation token for large files
    session = requests.Session()
    response = session.get(url, stream=True)

    # Handle the virus scan warning page for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        url = f"{url}&confirm={token}"
        response = session.get(url, stream=True)

    total = int(response.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=SRAF_FILENAME
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Saved to {out_path}")
    return out_path


def process_sraf_data(zip_path: Path, top100_ciks: set) -> pd.DataFrame:
    """
    Extract and filter SRAF daily data to top 100 firms.
    Returns a DataFrame with daily download counts per (cik, accession, date).
    """
    print("Processing SRAF zip file...")
    dfs = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        print(f"Found {len(csv_files)} CSV files in archive")

        for fname in tqdm(csv_files, desc="Processing files"):
            with zf.open(fname) as f:
                try:
                    df = pd.read_csv(
                        f,
                        dtype={"cik": str},
                        parse_dates=["date", "filing_date"],
                        low_memory=False,
                    )
                    # filter to top 100 firms immediately to keep memory low
                    df = df[df["cik"].isin(top100_ciks)]
                    if len(df) > 0:
                        dfs.append(df)
                except Exception as e:
                    print(f"  Warning: skipped {fname} ({e})")

    if not dfs:
        raise ValueError("No data found for top 100 firms in SRAF archive")

    result = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(result):,} rows for top 100 firms")
    return result


def enrich_with_firm_names(df: pd.DataFrame) -> pd.DataFrame:
    with open(TOP100_PATH) as f:
        firms = json.load(f)
    cik_to_name = {f["cik"]: f["name"] for f in firms}
    df["company_name"] = df["cik"].map(cik_to_name)
    return df


def build_edgar_daily(force: bool = False) -> pd.DataFrame:
    """
    Full pipeline: download → filter → enrich → save parquet.
    Returns the processed DataFrame.
    """
    out_path = PROCESSED_DIR / "edgar_daily_top100.parquet"

    if out_path.exists() and not force:
        print(f"Loading cached data from {out_path}")
        return pd.read_parquet(out_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    top100_ciks = load_top100_ciks()

    zip_path = download_sraf_data()
    df = process_sraf_data(zip_path, top100_ciks)
    df = enrich_with_firm_names(df)

    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    return df


if __name__ == "__main__":
    df = build_edgar_daily()
    print(df.head())
    print(df.dtypes)
    print(f"\nDate range: {df['date'].min()} — {df['date'].max()}")
    print(f"Unique firms: {df['cik'].nunique()}")
    print(f"Unique form types: {df['form'].value_counts().head(10)}")
