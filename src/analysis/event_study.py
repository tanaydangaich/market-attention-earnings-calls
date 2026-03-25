"""
Event study: abnormal EDGAR downloads around earnings calls.

Methodology (analogous to stock return event studies):
  1. For each earnings call, define an event window [-PRE, +POST] trading days
  2. Estimate a baseline download rate from a clean estimation window
     ([-BASELINE-PRE, -PRE-1] days before the call)
  3. Abnormal downloads (AD) = actual downloads - expected (baseline rate × days)
  4. Cumulative abnormal downloads (CAD) = sum(AD) over event window
  5. Regress CAD on tone scores, controlling for firm and year fixed effects

Key output:
  - event_study.parquet: one row per earnings call with AD time series
  - regression_results.json: core finding (tone → CAD)
  - crisis_comparison.parquet: pre/during/post 2008 crisis breakdown
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.config import (
    PROCESSED_DIR,
    EVENT_WINDOW_PRE,
    EVENT_WINDOW_POST,
    BASELINE_WINDOW,
    CRISIS_START,
    CRISIS_END,
)


# ── Baseline estimation ───────────────────────────────────────────────────────

def estimate_baseline(
    edgar_df: pd.DataFrame,
    cik: str,
    event_date: pd.Timestamp,
    baseline_window: int = BASELINE_WINDOW,
    pre_window: int = EVENT_WINDOW_PRE,
) -> float:
    """
    Estimate expected daily downloads for a firm before an earnings call.
    Uses the mean daily downloads in the baseline window, excluding the event window.

    Returns: expected downloads per day (float)
    """
    baseline_end = event_date - pd.Timedelta(days=pre_window + 1)
    baseline_start = baseline_end - pd.Timedelta(days=baseline_window)

    firm_data = edgar_df[
        (edgar_df["cik"] == cik)
        & (edgar_df["date"] >= baseline_start)
        & (edgar_df["date"] <= baseline_end)
    ]

    if firm_data.empty:
        return np.nan

    # aggregate to daily totals
    daily = firm_data.groupby("date")["nr_total"].sum()
    return daily.mean()


# ── Event window downloads ────────────────────────────────────────────────────

def get_event_window_downloads(
    edgar_df: pd.DataFrame,
    cik: str,
    event_date: pd.Timestamp,
    pre: int = EVENT_WINDOW_PRE,
    post: int = EVENT_WINDOW_POST,
) -> pd.DataFrame:
    """
    Get daily download totals for a firm in the event window.
    Returns DataFrame with columns: relative_day, date, actual_downloads
    """
    window_start = event_date - pd.Timedelta(days=pre)
    window_end = event_date + pd.Timedelta(days=post)

    firm_data = edgar_df[
        (edgar_df["cik"] == cik)
        & (edgar_df["date"] >= window_start)
        & (edgar_df["date"] <= window_end)
    ]

    if firm_data.empty:
        return pd.DataFrame()

    daily = firm_data.groupby("date")["nr_total"].sum().reset_index()
    daily.columns = ["date", "actual_downloads"]
    daily["relative_day"] = (daily["date"] - event_date).dt.days
    return daily


# ── Core event study ──────────────────────────────────────────────────────────

def compute_abnormal_downloads(
    edgar_df: pd.DataFrame,
    tone_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each earnings call in tone_df, compute:
      - baseline download rate
      - actual downloads in event window
      - abnormal downloads (actual - expected)
      - cumulative abnormal downloads (CAD) over [0, +30] and [0, +5]

    Returns one row per earnings call.
    """
    results = []

    for _, call in tone_df.iterrows():
        cik = str(call["cik"])
        event_date = pd.Timestamp(call["call_date"])

        baseline_rate = estimate_baseline(edgar_df, cik, event_date)
        window = get_event_window_downloads(edgar_df, cik, event_date)

        if window.empty or np.isnan(baseline_rate):
            continue

        window = window.copy()
        window["expected_downloads"] = baseline_rate
        window["abnormal_downloads"] = window["actual_downloads"] - baseline_rate

        # cumulative abnormal downloads over different windows
        post = window[window["relative_day"] >= 0]
        post_5 = window[(window["relative_day"] >= 0) & (window["relative_day"] <= 5)]
        post_30 = window[(window["relative_day"] >= 0) & (window["relative_day"] <= 30)]

        results.append({
            "cik": cik,
            "company_name": call.get("company_name", ""),
            "call_date": event_date,
            "fiscal_year": call.get("fiscal_year"),
            "fiscal_quarter": call.get("fiscal_quarter"),
            # tone scores
            "lm_tone": call.get("full_lm_tone"),
            "lm_negative_pct": call.get("full_lm_negative_pct"),
            "lm_uncertain_pct": call.get("full_lm_uncertain_pct"),
            "prepared_lm_tone": call.get("prepared_lm_tone"),
            "qa_lm_tone": call.get("qa_lm_tone"),
            "gpt_hedging_score": call.get("gpt_hedging_score"),
            "gpt_evasiveness_score": call.get("gpt_evasiveness_score"),
            # attention metrics
            "baseline_daily_downloads": baseline_rate,
            "cad_5": post_5["abnormal_downloads"].sum(),
            "cad_30": post_30["abnormal_downloads"].sum(),
            "cad_30_pct": (
                post_30["abnormal_downloads"].sum() / (baseline_rate * 30) * 100
                if baseline_rate > 0 else np.nan
            ),
            # is this a crisis period call?
            "is_crisis": (
                pd.Timestamp(CRISIS_START) <= event_date <= pd.Timestamp(CRISIS_END)
            ),
        })

    return pd.DataFrame(results)


# ── Regression analysis ───────────────────────────────────────────────────────

def run_regression(df: pd.DataFrame) -> dict:
    """
    Core regression: CAD_30 ~ tone + controls + firm FE + year FE

    Hypotheses:
      H1: lm_tone is negatively associated with CAD_30
          (more negative tone → more downloads)
      H2: lm_uncertain_pct is positively associated with CAD_30
          (more uncertainty language → more downloads)
      H3: effect is larger during crisis period
    """
    df = df.dropna(subset=["cad_30", "lm_tone", "lm_uncertain_pct"])
    df["year"] = pd.to_datetime(df["call_date"]).dt.year
    df["cik"] = df["cik"].astype(str)

    results = {}

    # Model 1: tone only
    m1 = smf.ols("cad_30 ~ lm_tone + lm_uncertain_pct", data=df).fit(cov_type="HC3")
    results["model_1_tone_only"] = {
        "coef_lm_tone": m1.params.get("lm_tone"),
        "pval_lm_tone": m1.pvalues.get("lm_tone"),
        "coef_lm_uncertain": m1.params.get("lm_uncertain_pct"),
        "pval_lm_uncertain": m1.pvalues.get("lm_uncertain_pct"),
        "r_squared": m1.rsquared,
        "n_obs": int(m1.nobs),
    }

    # Model 2: add firm and year fixed effects
    m2 = smf.ols(
        "cad_30 ~ lm_tone + lm_uncertain_pct + C(cik) + C(year)", data=df
    ).fit(cov_type="HC3")
    results["model_2_with_FE"] = {
        "coef_lm_tone": m2.params.get("lm_tone"),
        "pval_lm_tone": m2.pvalues.get("lm_tone"),
        "coef_lm_uncertain": m2.params.get("lm_uncertain_pct"),
        "pval_lm_uncertain": m2.pvalues.get("lm_uncertain_pct"),
        "r_squared": m2.rsquared,
        "n_obs": int(m2.nobs),
    }

    # Model 3: crisis interaction
    df["crisis_x_tone"] = df["is_crisis"].astype(int) * df["lm_tone"]
    m3 = smf.ols(
        "cad_30 ~ lm_tone + lm_uncertain_pct + is_crisis + crisis_x_tone + C(cik) + C(year)",
        data=df,
    ).fit(cov_type="HC3")
    results["model_3_crisis_interaction"] = {
        "coef_lm_tone": m3.params.get("lm_tone"),
        "coef_crisis_x_tone": m3.params.get("crisis_x_tone"),
        "pval_crisis_x_tone": m3.pvalues.get("crisis_x_tone"),
        "r_squared": m3.rsquared,
        "n_obs": int(m3.nobs),
    }

    return results


def compute_average_event_study(
    edgar_df: pd.DataFrame,
    tone_df: pd.DataFrame,
    n_tone_quantiles: int = 4,
) -> pd.DataFrame:
    """
    Compute average abnormal downloads by relative day and tone quartile.
    This produces the signature event study chart.
    """
    # assign tone quartiles
    tone_df = tone_df.copy()
    tone_df["tone_quartile"] = pd.qcut(
        tone_df["full_lm_tone"], q=n_tone_quantiles,
        labels=["Q1 (most negative)", "Q2", "Q3", "Q4 (most positive)"]
    )

    all_windows = []
    for _, call in tone_df.iterrows():
        window = get_event_window_downloads(
            edgar_df, str(call["cik"]), pd.Timestamp(call["call_date"])
        )
        if window.empty:
            continue
        baseline = estimate_baseline(edgar_df, str(call["cik"]), pd.Timestamp(call["call_date"]))
        if np.isnan(baseline) or baseline == 0:
            continue
        window["abnormal_pct"] = (window["actual_downloads"] - baseline) / baseline * 100
        window["tone_quartile"] = call["tone_quartile"]
        all_windows.append(window)

    if not all_windows:
        return pd.DataFrame()

    combined = pd.concat(all_windows, ignore_index=True)
    avg = (
        combined.groupby(["relative_day", "tone_quartile"])["abnormal_pct"]
        .mean()
        .reset_index()
    )
    return avg


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_event_study(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Full event study pipeline.
    Returns (event_study_df, regression_results).
    """
    out_path = PROCESSED_DIR / "event_study.parquet"
    reg_path = PROCESSED_DIR / "regression_results.json"

    if out_path.exists() and not force:
        print(f"Loading cached event study from {out_path}")
        df = pd.read_parquet(out_path)
        with open(reg_path) as f:
            reg = json.load(f)
        return df, reg

    edgar_path = PROCESSED_DIR / "edgar_daily_top100.parquet"
    tone_path = PROCESSED_DIR / "tone_scores.parquet"

    for p in [edgar_path, tone_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found — run earlier pipeline steps first")

    edgar_df = pd.read_parquet(edgar_path)
    edgar_df["date"] = pd.to_datetime(edgar_df["date"])

    tone_df = pd.read_parquet(tone_path)
    tone_df["call_date"] = pd.to_datetime(tone_df["call_date"])

    print(f"Computing abnormal downloads for {len(tone_df):,} earnings calls...")
    event_df = compute_abnormal_downloads(edgar_df, tone_df)

    print("Running regressions...")
    reg_results = run_regression(event_df)

    event_df.to_parquet(out_path, index=False)
    with open(reg_path, "w") as f:
        json.dump(reg_results, f, indent=2, default=str)

    print(f"Saved event study ({len(event_df):,} calls) to {out_path}")
    return event_df, reg_results


if __name__ == "__main__":
    df, reg = build_event_study()

    print("\n=== Regression Results ===")
    for model, res in reg.items():
        print(f"\n{model}:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Sample Event Study Data ===")
    print(df[["company_name", "call_date", "lm_tone", "cad_5", "cad_30"]].head(10))
    print(f"\nCorrelation (lm_tone vs cad_30): {df['lm_tone'].corr(df['cad_30']):.3f}")
