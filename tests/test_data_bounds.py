from unittest.mock import patch, MagicMock
from backend.data_engine import get_fundamental_metrics

@patch('backend.data_engine.yf.Ticker')
def test_fundamental_bounds_clamping(mock_ticker_class):
    # Setup mock to return negative and invalid values
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {
        "trailingPE": -5.5,
        "forwardPE": -10.0,
        "priceToBook": -1.2,
        "enterpriseToEbitda": -0.5,
        "marketCap": -1000000,
        "dividendYield": -0.05,
        "beta": -2.0,
        "sector": "Technology"
    }
    mock_ticker_class.return_value = mock_ticker_instance

    metrics = get_fundamental_metrics("FAKE")
    
    assert metrics["pe_ratio"] is None
    assert metrics["forward_pe"] is None
    assert metrics["price_to_book"] is None
    assert metrics["ev_to_ebitda"] is None
    assert metrics["market_cap"] is None
    assert metrics["dividend_yield"] is None
    # beta is allowed to be negative
    assert metrics["beta"] == -2.0

@patch('backend.data_engine.yf.Ticker')
def test_fundamental_bounds_valid(mock_ticker_class):
    # Setup mock to return valid values
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {
        "trailingPE": 15.5,
        "forwardPE": 10.0,
        "priceToBook": 1.2,
        "enterpriseToEbitda": 8.5,
        "marketCap": 1000000,
        "dividendYield": 0.05,
        "beta": 1.5,
        "sector": "Technology"
    }
    mock_ticker_class.return_value = mock_ticker_instance

    metrics = get_fundamental_metrics("FAKE2")
    
    assert metrics["pe_ratio"] == 15.5
    assert metrics["forward_pe"] == 10.0
    assert metrics["price_to_book"] == 1.2
    assert metrics["ev_to_ebitda"] == 8.5
    assert metrics["market_cap"] == 1000000
    assert metrics["dividend_yield"] == 0.05
    assert metrics["beta"] == 1.5
