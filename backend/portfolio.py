import pandas as pd
from typing import List, Dict

from backend.data_engine import fetch_data, get_fundamental_metrics
from backend.metrics import calculate_probabilities
from backend.backtester import compute_return_metrics


def _aggregate_shares_by_ticker(holdings: List[dict]) -> Dict[str, float]:
    """Sum shares across multiple lots of the same ticker (dollar-cost-averaging is normal)."""
    agg: Dict[str, float] = {}
    for h in holdings:
        tk = h["ticker"].upper().strip()
        agg[tk] = agg.get(tk, 0.0) + float(h["shares"])
    return agg


def analyze_portfolio(
    holdings: List[dict],
    horizon_days: int = 120,
    target_upside: float = 0.15,
    target_downside: float = -0.15,
    prob_model: str = "garch",
) -> dict:
    """
    holdings: list of {"ticker", "shares", "cost_basis"(optional), "purchase_date"(optional)}.

    Historical/forward metrics are computed by collapsing the portfolio into a single
    synthetic weighted daily-return series (today's market-value weights applied to each
    ticker's own historical returns), then reusing compute_return_metrics/calculate_probabilities
    exactly as they're used for a single ticker elsewhere in this codebase.
    """
    if not holdings:
        return {"status": "error", "message": "Portfolio has no holdings"}

    shares_by_ticker = _aggregate_shares_by_ticker(holdings)
    unique_tickers = list(shares_by_ticker.keys())

    price_series = {}
    excluded = []
    for tk in unique_tickers:
        try:
            data = fetch_data(tk, start="2020-01-01")
            if data.empty or "Close" not in data.columns:
                excluded.append({"ticker": tk, "reason": "No price data available"})
                continue
            price_series[tk] = data["Close"].dropna()
        except Exception as e:
            excluded.append({"ticker": tk, "reason": str(e)})

    if not price_series:
        return {"status": "error", "message": "Could not fetch price data for any holding", "excluded": excluded}

    # Current prices come from each ticker's own latest close, not the aligned/intersected frame.
    current_prices = {tk: float(s.iloc[-1]) for tk, s in price_series.items()}
    market_values = {tk: current_prices[tk] * shares_by_ticker[tk] for tk in price_series}
    total_market_value = sum(market_values.values())
    weights = {tk: mv / total_market_value for tk, mv in market_values.items()} if total_market_value > 0 else {}

    # Inner-join all series onto common trading days to build the weighted synthetic series.
    # This is a real simplification worth surfacing: it answers "how would today's exact
    # allocation have performed historically," not "how did my actual dollars perform given
    # real weight drift" - the frontend should label it accordingly.
    price_df = pd.DataFrame(price_series).dropna(how="any")
    history_start_date = None
    constraining_tickers = []
    if not price_df.empty:
        history_start_date = str(price_df.index[0].date())
        first_dates = {tk: s.dropna().index[0] for tk, s in price_series.items()}
        max_first_date = max(first_dates.values())
        constraining_tickers = [tk for tk, d in first_dates.items() if d == max_first_date]

    returns_df = price_df.pct_change().dropna(how="all")
    weight_vector = pd.Series({tk: weights.get(tk, 0.0) for tk in price_df.columns})
    portfolio_daily_returns = (returns_df * weight_vector).sum(axis=1)

    historical = None
    if len(portfolio_daily_returns) > 1:
        rm = compute_return_metrics(portfolio_daily_returns)
        historical = {
            "metrics": rm["metrics"],
            "history_start_date": history_start_date,
            "constraining_tickers": constraining_tickers,
        }

    forward = {"prob_success": 0.0, "prob_stop": 0.0, "exp_return": 0.0, "insufficient_history": True}
    if len(portfolio_daily_returns) >= 252:
        synthetic_price = (1 + portfolio_daily_returns).cumprod()
        probs = calculate_probabilities(
            synthetic_price,
            horizon_days=horizon_days,
            target_upside=target_upside,
            max_downside=target_downside,
            method=prob_model,
        )
        forward = {**probs, "insufficient_history": False}

    # Per-ticker cost-basis rollup (a ticker with any cost-basis-less lot can't get an exact P&L).
    cost_by_ticker: Dict[str, dict] = {}
    for h in holdings:
        tk = h["ticker"].upper().strip()
        if tk not in current_prices:
            continue
        entry = cost_by_ticker.setdefault(tk, {"cost_total": 0.0, "has_full_cost_basis": True})
        cost_basis = h.get("cost_basis")
        if cost_basis is not None:
            entry["cost_total"] += float(cost_basis) * float(h["shares"])
        else:
            entry["has_full_cost_basis"] = False

    holdings_breakdown = []
    total_cost_basis = 0.0
    portfolio_has_full_cost_basis = True
    for tk in unique_tickers:
        if tk not in current_prices:
            continue

        fundamentals = get_fundamental_metrics(tk)
        holding_probs = calculate_probabilities(
            price_series[tk],
            horizon_days=horizon_days,
            target_upside=target_upside,
            max_downside=target_downside,
            method=prob_model,
        )

        cost_info = cost_by_ticker.get(tk, {"cost_total": 0.0, "has_full_cost_basis": False})
        unrealized_pnl = None
        unrealized_pnl_pct = None
        if cost_info["has_full_cost_basis"] and cost_info["cost_total"] > 0:
            unrealized_pnl = market_values[tk] - cost_info["cost_total"]
            unrealized_pnl_pct = unrealized_pnl / cost_info["cost_total"]
            total_cost_basis += cost_info["cost_total"]
        else:
            portfolio_has_full_cost_basis = False

        holdings_breakdown.append({
            "ticker": tk,
            "shares": shares_by_ticker[tk],
            "current_price": current_prices[tk],
            "market_value": market_values[tk],
            "weight": weights.get(tk, 0.0),
            "sector": fundamentals.get("sector", "Unknown Sector"),
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "prob_success": holding_probs.get("prob_success", 0.0),
            "prob_stop": holding_probs.get("prob_stop", 0.0),
        })

    sector_allocation: Dict[str, float] = {}
    for h in holdings_breakdown:
        sector_allocation[h["sector"]] = sector_allocation.get(h["sector"], 0.0) + h["market_value"]

    total_unrealized_pnl = None
    total_unrealized_pnl_pct = None
    if portfolio_has_full_cost_basis and total_cost_basis > 0:
        total_unrealized_pnl = total_market_value - total_cost_basis
        total_unrealized_pnl_pct = total_unrealized_pnl / total_cost_basis

    return {
        "status": "success",
        "total_market_value": total_market_value,
        "total_cost_basis": total_cost_basis if portfolio_has_full_cost_basis else None,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
        "has_full_cost_basis": portfolio_has_full_cost_basis,
        "holdings": holdings_breakdown,
        "sector_allocation": sector_allocation,
        "historical": historical,
        "forward": forward,
        "excluded": excluded,
    }
