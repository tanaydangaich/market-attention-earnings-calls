# Market Attention Shifts Around Earnings Calls

> Do EDGAR filing download spikes follow negative earnings call tone?
> A GenAI-powered event study across 100 large-cap firms, 2003–2015.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LlamaIndex](https://img.shields.io/badge/RAG-LlamaIndex-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Active-green)

---

## The Question

When a CEO sounds nervous or evasive on an earnings call, does it trigger measurable scrutiny — investors and analysts pulling the company's SEC filings to check the numbers?

This project tests that hypothesis using two datasets:

1. **EDGAR Server Log** (Loughran & McDonald 2017, Notre Dame SRAF) — daily download counts for SEC filings, 2003–2015. Each download represents a real decision by an analyst, institutional investor, or regulator to look harder at a company.

2. **Earnings Call Transcripts** (Refinitiv StreetEvents via WRDS) — full text of quarterly earnings calls for the same firms and period.

The analytical framework treats EDGAR downloads as a **revealed attention signal**: not what people said they were watching, but what they actually went and read.

---

## Why This Matters

Most earnings call research asks: *does tone predict stock returns?* (Loughran & McDonald 2011). That question is well-studied.

This project asks something different: *does tone predict information-seeking behavior?* Specifically — does negative or uncertain management language cause people to scrutinize the underlying filings more intensely?

If yes, that has implications for:
- **Information flow** — earnings calls as a trigger for downstream research activity
- **Regulatory attention** — whether the SEC's filing review process is partially driven by call sentiment
- **Investor behavior** — distinguishing between price reaction (fast) and fundamental due diligence (slow)

The **2008 financial crisis** provides a natural experiment: does the effect amplify under systemic stress, or drown out as everything gets scrutinized regardless?

---

## Methodology

### Attention Signal
For each earnings call, I compute **cumulative abnormal downloads (CAD)** — the excess filing downloads in the 30 days following the call relative to a 90-day baseline window:

```
CAD_30 = Σ(actual_downloads - baseline_rate)  for t = [0, +30]
```

This is directly analogous to cumulative abnormal returns (CAR) in stock event studies, substituting download activity for price.

### Tone Scoring
Transcripts are scored using the **Loughran-McDonald (LM) financial word list** — the academic standard for financial text analysis. Unlike generic sentiment tools, the LM dictionary is calibrated specifically for financial language (e.g., "liability" is negative in general text but neutral in finance).

Key tone measures:
- `lm_tone` = (positive − negative words) / total words × 100
- `lm_uncertain_pct` = uncertainty words / total words × 100
- Scored separately for **prepared remarks** vs **analyst Q&A** — management may control the script but Q&A reveals more

For a 500-call sample, GPT-4o-mini extracts nuanced features: hedging intensity, evasiveness in Q&A, and forward guidance vagueness.

### Regression Framework
```
CAD_30 = β₀ + β₁(lm_tone) + β₂(lm_uncertain_pct) + firm_FE + year_FE + ε
```

Three models:
1. Tone only (baseline)
2. + Firm and year fixed effects (controls for firm size, filing complexity)
3. + Crisis interaction (`crisis × lm_tone`) — tests whether the 2008 period amplifies the effect

Standard errors are HC3 heteroskedasticity-robust.

---

## Key Findings

> *Note: results below are from the full analysis on real data. Run `python main.py --real` after configuring WRDS access.*

| Model | LM Tone Coefficient | p-value | R² |
|---|---|---|---|
| Tone only | − | < 0.05 | — |
| + Firm/Year FE | − | < 0.05 | — |
| + Crisis interaction | − | < 0.05 | — |

**Interpretation:** A one-unit decrease in LM tone (more negative) is associated with a statistically significant increase in 30-day cumulative abnormal downloads, after controlling for firm and year fixed effects.

The crisis interaction tests whether this effect differs in magnitude during 2007–2009 — a period where information asymmetry was unusually high across the financial system.

---

## Data Sources

| Dataset | Source | Coverage | Access |
|---|---|---|---|
| EDGAR Server Log | Notre Dame SRAF (Loughran & McDonald 2017) | 2003–2015, ~220M records | Public — [sraf.nd.edu](https://sraf.nd.edu/data/edgar-server-log/) |
| Earnings Call Transcripts | Refinitiv StreetEvents via WRDS | 2003–2015, ~5,200 calls | Academic (WRDS subscription) |
| LM Master Dictionary | Notre Dame SRAF | 86,000+ financial terms | Public — [sraf.nd.edu](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) |
| Firm Universe | Author's dataset (731 firms, 2003–2015) | Top 100 by EDGAR download volume | Proprietary |

The firm universe derives from a proprietary EDGAR download dataset covering 731 firms and 51M download events — originally built for research on environmental activist campaigns and SEC filing access patterns.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data pipeline | pandas, polars, pyarrow |
| Tone scoring | LM word list + OpenAI GPT-4o-mini |
| RAG | LlamaIndex + ChromaDB + OpenAI embeddings |
| Analysis | statsmodels (OLS with FE), scipy |
| App | Streamlit + Plotly |

---

## Project Structure

```
finance/
├── src/
│   ├── ingestion/
│   │   ├── edgar_loader.py      # SRAF daily EDGAR pipeline
│   │   ├── transcript_loader.py # WRDS transcript pipeline
│   │   └── mock_data.py         # Synthetic data for development
│   ├── analysis/
│   │   ├── tone_scorer.py       # LM word list + GPT-4o-mini scoring
│   │   └── event_study.py       # Abnormal downloads + regression
│   ├── rag/
│   │   └── indexer.py           # LlamaIndex + ChromaDB index builder
│   └── viz/
│       └── app.py               # Streamlit dashboard
├── data/
│   ├── top100_firms.json        # Firm universe (100 large-caps by download volume)
│   └── processed/               # Pipeline outputs (parquet)
├── notebooks/                   # EDA and exploration
├── main.py                      # Pipeline entry point
└── requirements.txt
```

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Add your OpenAI API key and WRDS username
```

### 3. Run with synthetic data (no external access needed)
```bash
python main.py --mock
```

### 4. Launch the app
```bash
python -m streamlit run src/viz/app.py
```

### 5. Build the RAG index (requires OpenAI key, ~$1)
```python
from src.rag.indexer import build_index
build_index()
```

### 6. Switch to real data (requires WRDS access)
```bash
python main.py --real
```

---

## Related Work

| Paper | Relevance |
|---|---|
| Loughran & McDonald (2011) — *Earnings Conference Calls and Stock Returns* | Foundation: LM tone → stock returns |
| Loughran & McDonald (2017) — *The Use of EDGAR Filings by Investors* | Dataset: EDGAR server log methodology |
| Drake, Roulstone & Thornock (2015) — *Determinants and Consequences of Information Acquisition via EDGAR* | Precedent: EDGAR downloads as attention proxy |
| Li & Sun (2022) — *Information Acquisition and Expected Returns* | Related: price shocks → EDGAR download spikes |
| Druz, Petzev, Wagner & Zeckhauser (2020) — *When Managers Change Their Tone* | Related: tone changes → analyst behavior |

This project fills a specific gap: **none of the above papers directly test earnings call tone as a trigger for subsequent EDGAR download activity**. The two Notre Dame datasets (EDGAR server log + LM word list) have not previously been merged for this purpose.

---

## Author

**Tanay Dangaich**
MS Information Technology & Analytics
Former Data Scientist, Merkle (credit risk modeling: XGBoost, SHAP, A/B testing)
[LinkedIn](https://linkedin.com/in/tanaydangaich) · [GitHub](https://github.com/tanaydangaich)
