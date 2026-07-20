# Sifter

Sifter is a quantitative equity/ETF screener and portfolio analysis tool. It ranks a
configurable stock/ETF universe using value, momentum, analyst-sentiment, and
volatility-model signals, layers in AI-generated macro sentiment via a local LLM, and lets
you save and analyze your own real portfolio against the same models. Everything runs
locally: a FastAPI backend, a SQLite database, and a single-file vanilla-JS frontend with
no build step.

## Features

**Market Screener** (main dashboard)
- Ranks the full universe on a blend of value score (P/E, P/B, EV/EBITDA), a
  detrended momentum z-score, analyst buy ratings, and a modeled probability of
  hitting your upside/downside targets within a chosen horizon.
- Selectable probability model: GARCH, GJR-GARCH, EGARCH, APARCH, TGARCH, GBM Monte
  Carlo, or historical empirical resampling.
- Adjustable strategy weights and Bayesian subjective-view overrides per ticker.
- AI macro sentiment: a local Ollama model scores each sector from -1 to +1 based on
  current VIX/10Y yield/SPY trend and optional live news search, and nudges rank
  scores accordingly.
- Per-ticker AI-generated web sentiment notes (yfinance news + DuckDuckGo search).
- "Ask AI for Final Pick" - has the LLM pick a top pick and a higher-risk reach pick
  from your top-ranked candidates.

**Universe**
- Built from the S&P 500, Dow Jones, Nasdaq-100, and Russell 1000 (scraped from
  Wikipedia), plus a curated set of low-fee Vanguard/iShares ETFs (VTI, VOO, VUG,
  VTV, VIG, VYM, VB, VXUS, IWM, IWF, IWD, SPLG, ITOT, IVV) and a dedicated
  **Defense & Aerospace** sector (ITA, PPA, XAR, SHLD, and primes like LMT, RTX,
  NOC, GD, LHX, HII, TXT, KTOS, AVAV) that's reclassified out of the generic
  "Industrials" bucket yfinance assigns them.
- Universe Manager tab to view the current universe and manually add/validate
  additional tickers.

**My Portfolio**
- Save your own real holdings (ticker, shares, cost basis, purchase date) under
  multiple named portfolios.
- Historical metrics (total/annualized return, Sharpe, Sortino, max drawdown)
  computed by backtesting today's exact allocation over its available joint history.
- Forward-looking metrics (probability of gain/loss, expected return) computed with
  the same probability models used by the screener, applied to the portfolio as a
  whole.
- Per-holding breakdown: current price, market value, weight, unrealized P&L, and
  sector allocation.

**Other tools**
- **Stat-Arb / Pairs**: cointegration test between two tickers, hedge ratio, half-life,
  and a vectorized pairs backtest.
- **Vectorized Backtesting**: a simple mean-reversion (z-score) backtest on a single
  ticker.
- **Portfolio Finder**: Monte Carlo search for the top-Sortino 5-stock baskets over
  the last month.
- **Historical Analysis**: browse past screener runs with to-date performance
  tracking, plus analyst-rating and AI macro-score history lookups.

## Architecture

- **Backend**: Python / FastAPI (`backend/`). Market data via `yfinance`, cached to
  SQLite (`backend/sifter.db`) and Parquet (`backend/history/parquet_cache/`).
  Screener runs and AI reports are saved as JSON under `backend/history/`.
- **Frontend**: a single file, `static/index.html` - vanilla JS, no npm/build step,
  Chart.js loaded from a CDN.
- **AI**: a local [Ollama](https://ollama.com) server, not a cloud API. Used for macro
  sentiment scoring, per-ticker sentiment notes, AI report generation, and the
  final-pick summarizer.

## Install

**Prerequisites**
- Python 3.10+
- [Ollama](https://ollama.com), installed and able to run locally. The app checks for
  Ollama at startup and **will not start without it**.

**1. Clone and install Python dependencies**

```bash
git clone https://github.com/hashtagcpt/sifter.git
cd sifter
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Pull an Ollama model**

By default the backend requires the model `llama3:70b` to be available (a large,
~40GB download):

```bash
ollama pull llama3:70b
ollama serve   # if it isn't already running as a background service
```

If you'd rather use a smaller local model, pull it instead (e.g. `ollama pull
llama3.1:8b`) and change `preferred_model` in `backend/main.py`'s
`check_ollama_status()` to match - the startup check only accepts the exact model
name configured there.

**3. Run the app**

```bash
python -m backend.main
```

This starts the FastAPI server (with auto-reload) on `http://localhost:8000` and
serves the frontend from `/`. Open that URL in a browser.

**4. Run the tests** (optional)

```bash
pytest tests/
```

Note: a handful of loose `test_*.py` scripts may exist at the project root outside
`tests/` from ad-hoc debugging - those are excluded from the repo by `.gitignore` and
aren't part of the real test suite.

## Notes

- All market data comes from `yfinance`; expect occasional rate-limiting or gaps for
  thinly-traded tickers.
- The database (`backend/sifter.db`), scraped universe/sector caches, and screener
  history are excluded from git (see `.gitignore`) - they're generated locally the
  first time you run the app.
- See the in-app **Documentation** menu (`Methodology & Math`, `Gaps &
  Recommendations`) for details on the scoring math and known limitations.
