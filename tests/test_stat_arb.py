import pytest
import pandas as pd
import numpy as np
from backend.stat_arb import calculate_cointegration, estimate_half_life, generate_z_scores

def test_test_cointegration():
    # Generate perfectly cointegrated series
    np.random.seed(42)
    x = np.cumsum(np.random.randn(100))
    y = 2.0 * x + np.random.randn(100)
    
    series_x = pd.Series(x)
    series_y = pd.Series(y)
    
    res = calculate_cointegration(series_y, series_x)
    
    # Since they are cointegrated, p-value should be small
    assert res['p_value'] < 0.05
    # Hedge ratio should be close to 2.0
    assert abs(res['hedge_ratio'] - 2.0) < 0.2

def test_estimate_half_life():
    np.random.seed(42)
    # Simulate an OU process (mean reverting)
    # dS_t = theta * (mu - S_t) * dt + sigma * dW_t
    spread = [0.0]
    theta = 0.1
    for _ in range(1000):
        # discrete approximation
        s_next = spread[-1] + theta * (0 - spread[-1]) + np.random.randn()
        spread.append(s_next)
        
    s_series = pd.Series(spread)
    hl = estimate_half_life(s_series)
    
    # Expected half-life is ln(2)/theta ~ 6.93
    assert not np.isnan(hl)
    assert not np.isinf(hl)
    assert 4.0 < hl < 10.0

def test_generate_z_scores():
    np.random.seed(42)
    spread = pd.Series(np.random.randn(100))
    z = generate_z_scores(spread, window=10)
    
    # First 9 values should be NaN
    assert pd.isna(z.iloc[8])
    assert not pd.isna(z.iloc[9])
