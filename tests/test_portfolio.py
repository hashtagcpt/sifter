import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from backend.portfolio import analyze_portfolio


def _price_series(n=300, start_price=100.0, seed=1, end="2026-06-01"):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = start_price * np.exp(np.cumsum(returns))
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Close": prices}, index=idx)


FUNDAMENTALS_BY_TICKER = {
    "AAPL": {"sector": "Technology", "pe_ratio": 30.0},
    "MSFT": {"sector": "Technology", "pe_ratio": 32.0},
    "LMT": {"sector": "Defense & Aerospace", "pe_ratio": 18.0},
}


def _fake_fetch_data(ticker, start="2020-01-01", end=None):
    if ticker == "BADTICKER":
        return pd.DataFrame()
    if ticker == "NEWCO":
        return _price_series(n=60, seed=99)  # short history - constrains alignment
    return _price_series(n=300, seed=hash(ticker) % 1000)


def _fake_get_fundamental_metrics(ticker):
    return FUNDAMENTALS_BY_TICKER.get(ticker, {"sector": "Unknown Sector"})


@patch("backend.portfolio.get_fundamental_metrics", side_effect=_fake_get_fundamental_metrics)
@patch("backend.portfolio.fetch_data", side_effect=_fake_fetch_data)
def test_analyze_portfolio_basic_two_holdings(mock_fetch, mock_funds):
    holdings = [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 100.0, "purchase_date": "2024-01-01"},
        {"ticker": "MSFT", "shares": 5, "cost_basis": 200.0, "purchase_date": "2024-01-01"},
    ]
    result = analyze_portfolio(holdings, prob_model="historical")

    assert result["status"] == "success"
    assert len(result["holdings"]) == 2
    assert result["excluded"] == []
    assert result["total_market_value"] > 0
    assert abs(sum(h["weight"] for h in result["holdings"]) - 1.0) < 1e-6
    assert result["historical"] is not None
    assert "sharpe_ratio" in result["historical"]["metrics"]
    assert result["forward"]["insufficient_history"] is False
    assert result["sector_allocation"]["Technology"] > 0


@patch("backend.portfolio.get_fundamental_metrics", side_effect=_fake_get_fundamental_metrics)
@patch("backend.portfolio.fetch_data", side_effect=_fake_fetch_data)
def test_analyze_portfolio_excludes_failed_ticker_without_failing(mock_fetch, mock_funds):
    holdings = [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 100.0, "purchase_date": None},
        {"ticker": "BADTICKER", "shares": 3, "cost_basis": 50.0, "purchase_date": None},
    ]
    result = analyze_portfolio(holdings, prob_model="historical")

    assert result["status"] == "success"
    assert len(result["holdings"]) == 1
    assert result["holdings"][0]["ticker"] == "AAPL"
    assert len(result["excluded"]) == 1
    assert result["excluded"][0]["ticker"] == "BADTICKER"


@patch("backend.portfolio.get_fundamental_metrics", side_effect=_fake_get_fundamental_metrics)
@patch("backend.portfolio.fetch_data", side_effect=_fake_fetch_data)
def test_analyze_portfolio_all_tickers_fail_returns_error(mock_fetch, mock_funds):
    holdings = [{"ticker": "BADTICKER", "shares": 3, "cost_basis": 50.0, "purchase_date": None}]
    result = analyze_portfolio(holdings)
    assert result["status"] == "error"


@patch("backend.portfolio.get_fundamental_metrics", side_effect=_fake_get_fundamental_metrics)
@patch("backend.portfolio.fetch_data", side_effect=_fake_fetch_data)
def test_analyze_portfolio_reports_constraining_ticker_for_short_history(mock_fetch, mock_funds):
    holdings = [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 100.0, "purchase_date": None},
        {"ticker": "NEWCO", "shares": 10, "cost_basis": 20.0, "purchase_date": None},
    ]
    result = analyze_portfolio(holdings, prob_model="historical")

    assert result["status"] == "success"
    assert result["historical"] is not None
    assert "NEWCO" in result["historical"]["constraining_tickers"]
    # Combined history is limited to NEWCO's short window - forward metrics should be flagged.
    assert result["forward"]["insufficient_history"] is True


@patch("backend.portfolio.get_fundamental_metrics", side_effect=_fake_get_fundamental_metrics)
@patch("backend.portfolio.fetch_data", side_effect=_fake_fetch_data)
def test_analyze_portfolio_null_cost_basis_omits_pnl(mock_fetch, mock_funds):
    holdings = [
        {"ticker": "AAPL", "shares": 10, "cost_basis": None, "purchase_date": None},
        {"ticker": "MSFT", "shares": 5, "cost_basis": 200.0, "purchase_date": None},
    ]
    result = analyze_portfolio(holdings, prob_model="historical")

    aapl = next(h for h in result["holdings"] if h["ticker"] == "AAPL")
    msft = next(h for h in result["holdings"] if h["ticker"] == "MSFT")

    assert aapl["unrealized_pnl"] is None
    assert msft["unrealized_pnl"] is not None
    # Portfolio-level total is undefined when any holding lacks a cost basis.
    assert result["has_full_cost_basis"] is False
    assert result["total_unrealized_pnl"] is None


@patch("backend.portfolio.get_fundamental_metrics", side_effect=_fake_get_fundamental_metrics)
@patch("backend.portfolio.fetch_data", side_effect=_fake_fetch_data)
def test_analyze_portfolio_aggregates_multiple_lots_same_ticker(mock_fetch, mock_funds):
    holdings = [
        {"ticker": "AAPL", "shares": 5, "cost_basis": 100.0, "purchase_date": "2023-01-01"},
        {"ticker": "AAPL", "shares": 5, "cost_basis": 200.0, "purchase_date": "2024-01-01"},
    ]
    result = analyze_portfolio(holdings, prob_model="historical")

    assert len(result["holdings"]) == 1
    assert result["holdings"][0]["shares"] == 10
    # cost_total should be 5*100 + 5*200 = 1500
    expected_cost = 5 * 100.0 + 5 * 200.0
    assert result["total_cost_basis"] == pytest.approx(expected_cost)


def test_analyze_portfolio_empty_holdings_returns_error():
    result = analyze_portfolio([])
    assert result["status"] == "error"
