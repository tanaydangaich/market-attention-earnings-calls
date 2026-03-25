"""
Market Attention Shifts Around Earnings Calls
Full pipeline runner.

Usage:
  python main.py --mock        # run with synthetic data (no WRDS needed)
  python main.py --real        # run with real SRAF + WRDS data
  python main.py --app         # launch Streamlit app (needs processed data)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_mock_pipeline():
    print("\n=== Step 1: Generating mock data ===")
    from src.ingestion.mock_data import build_mock_data
    edgar_df, transcripts_df = build_mock_data()

    print("\n=== Step 2: Tone scoring ===")
    from src.analysis.tone_scorer import build_tone_scores
    tone_df = build_tone_scores(run_gpt=False)

    print("\n=== Step 3: Event study ===")
    from src.analysis.event_study import build_event_study
    event_df, reg_results = build_event_study()

    print("\n=== Pipeline complete ===")
    print(f"Transcripts scored: {len(tone_df):,}")
    print(f"Earnings calls in event study: {len(event_df):,}")
    print(f"\nCore finding (tone → 30-day abnormal downloads):")
    m2 = reg_results.get("model_2_with_FE", {})
    coef = m2.get("coef_lm_tone")
    pval = m2.get("pval_lm_tone")
    if coef is not None:
        direction = "negative" if coef < 0 else "positive"
        sig = "significant" if pval < 0.05 else "not significant"
        print(f"  lm_tone coefficient: {coef:.4f} ({direction}, p={pval:.4f}, {sig})")
    print("\nRun `python main.py --app` to launch the Streamlit dashboard.")


def run_real_pipeline():
    print("\n=== Step 1: Downloading SRAF daily EDGAR data ===")
    from src.ingestion.edgar_loader import build_edgar_daily
    build_edgar_daily()

    print("\n=== Step 2: Pulling WRDS transcripts ===")
    from src.ingestion.transcript_loader import build_transcripts
    build_transcripts()

    print("\n=== Step 3: Tone scoring ===")
    from src.analysis.tone_scorer import build_tone_scores
    build_tone_scores(run_gpt=True)

    print("\n=== Step 4: Event study ===")
    from src.analysis.event_study import build_event_study
    build_event_study()

    print("\nPipeline complete. Run `python main.py --app` to launch the dashboard.")


def run_app():
    subprocess.run(
        ["streamlit", "run", str(ROOT / "src" / "viz" / "app.py")],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Attention Shifts pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Run with synthetic data")
    group.add_argument("--real", action="store_true", help="Run with real SRAF + WRDS data")
    group.add_argument("--app", action="store_true", help="Launch Streamlit app")
    args = parser.parse_args()

    if args.mock:
        run_mock_pipeline()
    elif args.real:
        run_real_pipeline()
    elif args.app:
        run_app()
