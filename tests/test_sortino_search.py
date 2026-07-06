import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from backend.backtester import find_top_sortino_baskets

@patch("backend.backtester.yf.download")
@patch("backend.data_engine.build_universe")
def test_find_top_sortino_baskets(mock_universe, mock_download):
    # Mock universe
    mock_universe.return_value = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    
    # Mock data with 21 days (approx 1 month)
    dates = pd.date_range("2020-01-01", periods=21)
    
    # Create mock prices where some stocks go up linearly and some go down
    # To maximize sortino, the algorithm should pick the stocks that go up without down days (downside=0 -> Sortino=inf / very high)
    # But let's give them some small down days so sortino is computable.
    np.random.seed(42)
    prices = pd.DataFrame(index=dates, columns=mock_universe.return_value)
    for col in prices.columns:
        # random walk with positive drift
        returns = np.random.normal(0.001, 0.01, size=21)
        prices[col] = (1 + returns).cumprod() * 100
        
    prices.columns = pd.MultiIndex.from_product([['Close'], prices.columns])
    mock_download.return_value = prices
    
    baskets = find_top_sortino_baskets(basket_size=2, num_baskets=3, sample_size=10, iterations=10)
    
    assert len(baskets) == 3
    assert len(baskets[0]["tickers"]) == 2
    assert "sortino_ratio" in baskets[0]["metrics"]
    
    # Ensure they are non-overlapping
    all_tickers = []
    for b in baskets:
        all_tickers.extend(b["tickers"])
    assert len(all_tickers) == len(set(all_tickers)) == 6
