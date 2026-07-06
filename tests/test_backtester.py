import pytest
import pandas as pd
import numpy as np
from backend.backtester import run_vectorized_backtest

def test_run_vectorized_backtest():
    prices = pd.Series([100, 105, 100, 95, 100])
    
    # We enter a short position at index 1 (price 105) and long at index 3 (price 95)
    # Signals are generated at t, traded at t+1.
    # At t=0 (p=100), signal=-1. t=1 (p=105), position=-1. return=105/100-1 = 0.05. strategy_return = -1 * 0.05 = -0.05
    # wait, the backtester shifts the signal. position at t=1 is signal at t=0.
    
    signals = pd.Series([-1, 0, 1, 0, 0])
    
    # Let's use 0 transaction cost for simplicity
    res = run_vectorized_backtest(prices, signals, transaction_cost=0.0, borrow_cost_annual=0.0)
    
    assert "metrics" in res
    assert "timeseries" in res
    
    m = res["metrics"]
    
    # t=1: price=105, ret=+5%, pos=-1 -> strat_ret = -5%
    # t=2: price=100, ret=-4.76%, pos=0 -> strat_ret = 0%
    # t=3: price=95, ret=-5%, pos=1 -> strat_ret = -5%
    # t=4: price=100, ret=+5.26%, pos=0 -> strat_ret = 0%
    
    # cum return = (1 - 0.05) * (1 - 0) * (1 - 0.05) * (1 - 0) = 0.95 * 0.95 = 0.9025
    # So total_return ~ -0.0975
    
    assert abs(m["total_return"] - (-0.0975)) < 1e-4

def test_backtest_with_costs():
    prices = pd.Series([100, 101, 102])
    signals = pd.Series([1, 1, 1])
    
    res = run_vectorized_backtest(prices, signals, transaction_cost=0.01, borrow_cost_annual=0.0)
    m = res["metrics"]
    
    # t=1: price=101, ret=1%, pos=1. Trade at t=1 is pos(t=1)-pos(t=0) = 1-0 = 1. cost = 0.01.
    # strat_ret = 1% - 1% = 0.0
    # t=2: price=102, ret=~1%, pos=1. Trade=0. cost=0.
    # strat_ret = ~1%
    
    assert m["total_return"] > 0
    assert m["trades_count"] == 1.0
