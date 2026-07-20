import time
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")
from backend.metrics import detrend_and_zscore, calculate_probabilities
from backend.data_engine import build_universe, load_sector_index

TRANSACTION_COST = 0.002   # 0.2% round-trip (0.1% each side)
PORTFOLIO_SIZE   = 3
CORR_THRESHOLD   = 0.4

# ── Progress tracking (read by /api/backtest_progress) ───────────────────────
_progress: dict = {
    "running": False,
    "step": 0,
    "total_steps": 0,
    "start_time": 0.0,
}

def get_progress() -> dict:
    p = _progress
    elapsed = time.time() - p["start_time"] if p["running"] else 0.0
    rate = p["step"] / elapsed if elapsed > 0 and p["step"] > 0 else None
    eta  = (p["total_steps"] - p["step"]) / rate if rate else None
    pct  = p["step"] / p["total_steps"] if p["total_steps"] > 0 else 0.0
    return {
        "running":         p["running"],
        "step":            p["step"],
        "total_steps":     p["total_steps"],
        "pct":             round(pct * 100, 1),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds":     round(eta, 1) if eta is not None else None,
    }


# ── Bootstrap CI on backtest metrics ─────────────────────────────────────────

def bootstrap_backtest_ci(
    weekly_returns: list,
    n_bootstrap: int = 500,
    confidence: float = 0.90,
) -> dict:
    """
    Resample the observed weekly return sequence with replacement to build
    confidence intervals around total return, Sharpe, and max drawdown.

    Note: iid bootstrap ignores serial correlation; for a fun project this
    gives a useful first-order uncertainty estimate.
    """
    arr = np.array(weekly_returns, dtype=float)
    n   = len(arr)
    if n < 4:
        return {}

    rng = np.random.default_rng()
    total_rets, sharpes, drawdowns = [], [], []

    for _ in range(n_bootstrap):
        sample   = rng.choice(arr, size=n, replace=True)
        equity   = np.cumprod(1.0 + sample) * 1000.0
        tot      = equity[-1] / 1000.0 - 1.0
        eq_s     = pd.Series(equity)
        dd       = float((eq_s / eq_s.cummax() - 1).min())
        std      = sample.std()
        sharpe   = float(sample.mean() / std * np.sqrt(52)) if std > 0 else 0.0
        total_rets.append(tot)
        sharpes.append(sharpe)
        drawdowns.append(dd)

    alpha = 1.0 - confidence
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    def ci(v):
        return {"lower": round(float(np.percentile(v, lo)), 4),
                "upper": round(float(np.percentile(v, hi)), 4)}

    return {
        "confidence": confidence,
        "total_return":  ci(total_rets),
        "sharpe_ratio":  ci(sharpes),
        "max_drawdown":  ci(drawdowns),
    }


# ── Risk metrics ──────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, periods_per_year: int = 52) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

def _sortino(returns: pd.Series, periods_per_year: int = 52) -> float:
    down = returns[returns < 0]
    if len(down) < 2 or down.std() == 0:
        return 0.0
    return float(returns.mean() / down.std() * np.sqrt(periods_per_year))

def _calmar(total_return: float, max_drawdown: float, period_years: float) -> float:
    if max_drawdown == 0 or period_years == 0:
        return 0.0
    ann = (1 + total_return) ** (1 / period_years) - 1
    return float(ann / abs(max_drawdown))


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_custom_backtest(
    max_size: int   = 50,
    horizon_days: int = 120,
    target_upside: float = 0.15,
    max_downside: float  = -0.15,
    period_years: int    = 1,
) -> dict:

    universe = build_universe(max_size=max_size)
    if not universe:
        universe = ["AAPL", "MSFT", "GOOGL"]

    all_tickers = universe + ["SPY"]
    raw  = yf.download(all_tickers, period=f"{period_years + 2}y",
                       auto_adjust=True, progress=False)["Close"]
    data = raw.ffill()           # forward-fill only — no bfill to avoid look-ahead

    spy_series = data["SPY"] if "SPY" in data.columns else None
    universe   = [tk for tk in universe if tk in data.columns]

    total_days = len(data)
    if total_days < 252:
        return {"error": "Not enough historical data to backtest."}

    backtest_start_idx = max(total_days - (period_years * 252), 252)
    total_steps        = max(1, (total_days - backtest_start_idx - 20) // 20)

    # Initialise progress
    _progress.update({"running": True, "step": 0,
                       "total_steps": total_steps, "start_time": time.time()})

    capital     = 1000.0
    spy_entry   = float(spy_series.iloc[backtest_start_idx]) if spy_series is not None else None
    spy_capital = 1000.0

    equity_curve   = [{"date": str(data.index[backtest_start_idx - 1].date()),
                        "capital": capital, "spy": spy_capital}]
    trades         = []
    weekly_returns = []

    # Initialize Bayesian sector weights
    sector_idx = load_sector_index()
    all_sectors = set(sector_idx.values()) | {"Unknown Sector"}
    sector_priors = {sec: 1.0 for sec in all_sectors}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(backtest_start_idx, total_days - 20, 20):
            _progress["step"] += 1

            hist   = data.iloc[:i]
            recent = hist[universe].iloc[-120:].pct_change().dropna()
            corr   = recent.corr()

            def score_ticker(tk, _h=hist):
                prices = _h[tk].dropna()
                if len(prices) < 252:
                    return None
                z = detrend_and_zscore(prices, window=252)
                zv = float(z.iloc[-1]) if not pd.isna(z.iloc[-1]) else 0.0
                probs = calculate_probabilities(
                    prices, horizon_days=horizon_days,
                    target_upside=target_upside, max_downside=max_downside,
                    n_paths=100,
                )
                return {
                    "ticker":       tk,
                    "raw_rank_score":   probs["prob_success"] * 10.0 - zv * 0.5,
                    "prob_success": probs["prob_success"],
                    "z_score":      zv,
                }

            scored = [r for r in pool.map(score_ticker, universe) if r]
            
            # Apply Bayesian sector prior weighting
            for item in scored:
                sec = sector_idx.get(item["ticker"], "Unknown Sector")
                prior = sector_priors.get(sec, 1.0)
                item["raw_rank_score"] *= prior
                
            scored.sort(key=lambda x: x["raw_rank_score"])
            n = len(scored)
            for idx, res in enumerate(scored):
                res["rank_score"] = (idx / max(1, n - 1)) * 100.0 if n > 0 else 50.0
                
            scored.sort(key=lambda x: x["rank_score"], reverse=True)

            if not scored:
                if spy_series is not None and spy_entry:
                    spy_capital = 1000.0 * float(spy_series.iloc[i + 20]) / spy_entry
                equity_curve.append({"date": str(data.index[i + 20].date()),
                                      "capital": capital, "spy": spy_capital})
                continue

            # Greedy uncorrelated portfolio selection
            portfolio = []
            for item in scored:
                if len(portfolio) >= PORTFOLIO_SIZE:
                    break
                tk = item["ticker"]
                if any(
                    tk in corr.columns and sel in corr.columns
                    and corr.loc[tk, sel] > CORR_THRESHOLD
                    for sel in portfolio
                ):
                    continue
                portfolio.append(tk)

            # Fill remaining slots if uncorrelated picks run dry
            for item in scored:
                if len(portfolio) >= PORTFOLIO_SIZE:
                    break
                if item["ticker"] not in portfolio:
                    portfolio.append(item["ticker"])

            # Probability-weighted position sizing
            prob_map = {item["ticker"]: max(item["prob_success"], 0.01) for item in scored}
            raw_w    = {tk: prob_map.get(tk, 0.01) for tk in portfolio}
            tot_w    = sum(raw_w.values())
            weights  = {tk: w / tot_w for tk, w in raw_w.items()}

            # Dynamic Volatility Targeting based on SPY
            exposure = 1.0
            if spy_series is not None and i >= 60:
                spy_recent = spy_series.iloc[i-60:i].pct_change().dropna()
                spy_vol = spy_recent.std() * np.sqrt(252)
                # Scale exposure down when volatility exceeds 15% (0.15)
                # Reaches 0 exposure at 30% vol (0.30)
                exposure = max(0.0, min(1.0, 1.0 - (spy_vol - 0.15) / 0.15))

            port_ret   = 0.0
            trade_logs = []
            sector_returns = {}
            for tk in portfolio:
                try:
                    entry = float(data[tk].iloc[i])
                    exit_ = float(data[tk].iloc[i + 20])
                    ret   = (exit_ / entry - 1.0 - TRANSACTION_COST) if entry > 0 else 0.0
                    w     = weights[tk]
                    port_ret += ret * w
                    trade_logs.append({"ticker": tk, "weight": round(w, 3),
                                        "entry": entry, "exit": exit_, "return": ret})
                    sec = sector_idx.get(tk, "Unknown Sector")
                    if sec not in sector_returns:
                        sector_returns[sec] = []
                    sector_returns[sec].append(ret)
                except Exception:
                    pass

            # Apply exposure targeting (remaining capital is in cash, earning 0%)
            port_ret *= exposure

            # Bayesian Update of Sector Priors
            if spy_series is not None and spy_entry:
                spy_entry_step = float(spy_series.iloc[i])
                spy_exit_step = float(spy_series.iloc[i + 20])
                spy_ret = spy_exit_step / spy_entry_step - 1.0
                for sec, rets in sector_returns.items():
                    sec_avg_ret = float(np.mean(rets))
                    new_prior = sector_priors[sec] * (1.0 + 5.0 * (sec_avg_ret - spy_ret))
                    sector_priors[sec] = max(0.5, min(2.0, new_prior))

            capital *= (1.0 + port_ret)
            weekly_returns.append(port_ret)

            if spy_series is not None and spy_entry:
                spy_capital = 1000.0 * float(spy_series.iloc[i + 20]) / spy_entry

            equity_curve.append({"date": str(data.index[i + 20].date()),
                                   "capital": capital, "spy": spy_capital})
            trades.append({"date": str(data.index[i].date()),
                            "portfolio": trade_logs,
                            "weekly_return": port_ret,
                            "capital": capital})

    _progress["running"] = False

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_return    = capital / 1000.0 - 1.0
    equity_s        = pd.Series([e["capital"] for e in equity_curve])
    max_drawdown    = float((equity_s / equity_s.cummax() - 1).min())
    ret_series      = pd.Series(weekly_returns)
    sharpe          = _sharpe(ret_series)
    sortino         = _sortino(ret_series)
    calmar          = _calmar(total_return, max_drawdown, period_years)
    ann_return      = float((1 + total_return) ** (1 / period_years) - 1) if period_years > 0 else total_return
    win_rate        = float(np.mean([r > 0 for r in weekly_returns])) if weekly_returns else 0.0
    spy_total_ret   = (spy_capital / 1000.0 - 1.0) if spy_entry else None

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    ci = bootstrap_backtest_ci(weekly_returns, n_bootstrap=500, confidence=0.90)

    return {
        "status":           "success",
        "starting_capital": 1000.0,
        "ending_capital":   capital,
        "total_return":     total_return,
        "annualised_return": ann_return,
        "max_drawdown":     max_drawdown,
        "sharpe_ratio":     sharpe,
        "sortino_ratio":    sortino,
        "calmar_ratio":     calmar,
        "win_rate":         win_rate,
        "avg_trade_return": float(ret_series.mean()) if len(ret_series) else 0.0,
        "trade_count":      len(trades),
        "spy_total_return": spy_total_ret,
        "bootstrap_ci":     ci,
        "equity_curve":     equity_curve,
        "trades":           trades,
    }
