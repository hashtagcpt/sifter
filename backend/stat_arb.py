import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

def calculate_cointegration(series_y: pd.Series, series_x: pd.Series) -> dict:
    """
    Perform Engle-Granger cointegration test on two price series.
    Returns the p-value, hedge ratio (beta), and the spread series.
    """
    # Align data
    df = pd.concat([series_y, series_x], axis=1).dropna()
    if len(df) < 30:
        return {"p_value": 1.0, "hedge_ratio": 0.0, "spread": pd.Series(dtype=float)}
        
    y = df.iloc[:, 0]
    x = df.iloc[:, 1]
    
    # OLS regression to find hedge ratio
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = y - hedge_ratio * x
    
    # Cointegration test
    score, p_value, _ = coint(y, x)
    
    return {
        "p_value": p_value,
        "hedge_ratio": hedge_ratio,
        "spread": spread
    }

def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimates the half-life of mean reversion from an Ornstein-Uhlenbeck process.
    """
    if len(spread) < 30:
        return np.nan
        
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2).fit()
    
    theta = -model.params.iloc[1]
    if theta <= 0:
        return np.inf  # Not mean-reverting
        
    half_life = np.log(2) / theta
    return half_life

def generate_z_scores(spread: pd.Series, window: int = 21) -> pd.Series:
    """
    Generate rolling Z-scores for the spread.
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    z_score = (spread - rolling_mean) / rolling_std
    return z_score
