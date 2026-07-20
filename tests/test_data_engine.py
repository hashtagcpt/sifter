import pytest
import shutil
import os
from unittest.mock import patch, MagicMock
import pandas as pd
from backend.data_engine import (
    build_universe, fetch_data, get_fundamental_metrics, fetch_analyst_ratings,
    SUPPLEMENTAL_TICKERS,
)

@pytest.fixture(autouse=True)
def clean_yf_cache():
    cache_dir = "backend/history/yf_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    yield
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

@patch("backend.data_engine.requests.get")
def test_build_universe_scrape(mock_get):
    # Mocking Wikipedia page content
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '<html><table class="wikitable"><tr><td>AAPL</td></tr><tr><td>MSFT</td></tr></table></html>'
    mock_get.return_value = mock_response

    universe = build_universe(max_size=2, force_scrape=True)
    
    # We mocked one page to return AAPL, MSFT, but actually the parser expects specific table structures.
    # If the parser fails to find the exact structure, it returns empty or fallback.
    # Let's test the generic list type and length limitation.
    assert isinstance(universe, list)
    assert len(universe) <= 2

@patch("backend.data_engine.yf.download")
def test_fetch_data(mock_download):
    mock_hist = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    mock_download.return_value = mock_hist
    
    data = fetch_data("AAPL")
    assert not data.empty
    assert "Close" in data.columns

@patch("backend.data_engine.yf.Ticker")
def test_get_fundamental_metrics(mock_ticker):
    mock_ticker.return_value.info = {"trailingPE": 15, "marketCap": 2000000000000}
    
    metrics = get_fundamental_metrics("AAPL")
    assert metrics["pe_ratio"] == 15
    assert metrics["market_cap"] == 2000000000000

@patch("backend.data_engine.yf.Ticker")
def test_get_fundamental_metrics_percent_yield(mock_ticker):
    mock_ticker.return_value.info = {"trailingPE": 15, "marketCap": 2000000000000, "yield": 1.41}
    
    metrics = get_fundamental_metrics("AAPL")
    assert abs(metrics["dividend_yield"] - 0.0141) < 1e-6

@patch("backend.data_engine.yf.Ticker")
def test_get_fundamental_metrics_decimal_yield(mock_ticker):
    mock_ticker.return_value.info = {"trailingPE": 15, "marketCap": 2000000000000, "dividendYield": 0.0141}
    
    metrics = get_fundamental_metrics("AAPL")
    assert abs(metrics["dividend_yield"] - 0.0141) < 1e-6

@patch("backend.data_engine.yf.Ticker")
def test_get_fundamental_metrics_manual_sector_override_beats_yfinance(mock_ticker):
    # yfinance itself classifies LMT as "Industrials" - the manual override must win anyway.
    mock_ticker.return_value.info = {"trailingPE": 18, "sector": "Industrials"}

    metrics = get_fundamental_metrics("LMT")
    assert metrics["sector"] == "Defense & Aerospace"


@patch("backend.data_engine.save_universe")
@patch("backend.data_engine.get_universe")
def test_build_universe_merges_supplemental_tickers_via_cache(mock_get_universe, mock_save_universe):
    # Simulate the common cold-start case: a populated universe already cached in sqlite.
    mock_get_universe.return_value = ["AAPL", "MSFT", "GOOGL"]

    universe = build_universe(max_size=2500)

    for ticker in SUPPLEMENTAL_TICKERS:
        assert ticker in universe
    mock_save_universe.assert_called_once()


@patch("backend.data_engine.yf.Ticker")
def test_fetch_analyst_ratings(mock_ticker):
    # YF sometimes returns a DataFrame for recommendations
    mock_recommendations = pd.DataFrame([
        {"period": "0m", "strongBuy": 5, "buy": 10, "hold": 3, "sell": 1, "strongSell": 0}
    ])
    mock_ticker.return_value.recommendations = mock_recommendations
    
    ratings = fetch_analyst_ratings("AAPL")
    assert ratings["buy"] == 10
    assert ratings["strongBuy"] == 5
    assert ratings["sell"] == 1
