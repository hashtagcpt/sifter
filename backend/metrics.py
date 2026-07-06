import numpy as np
import pandas as pd
from scipy.signal import detrend
import statsmodels.api as sm
from arch import arch_model
import yfinance as yf

def calculate_rv(prices: pd.Series, window: int = 21) -> float:
    if len(prices) < window:
        return 0.0
    returns = prices.pct_change().dropna()
    return float(returns.tail(window).std() * np.sqrt(252))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    if len(close) < window + 1:
        return 0.0
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = df['tr'].rolling(window=window).mean()
    return float(atr.iloc[-1])

def calculate_hv_rank(prices: pd.Series, window: int = 21, lookback: int = 252) -> float:
    if len(prices) < lookback + window:
        return 0.0
    returns = prices.pct_change().dropna()
    rolling_rv = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_rv = rolling_rv.dropna().tail(lookback)
    if rolling_rv.empty:
        return 0.0
    current_rv = rolling_rv.iloc[-1]
    min_rv = rolling_rv.min()
    max_rv = rolling_rv.max()
    if max_rv == min_rv:
        return 0.0
    rank = (current_rv - min_rv) / (max_rv - min_rv) * 100.0
    return float(rank)

def fetch_current_iv(ticker_symbol: str) -> float:
    try:
        tk = yf.Ticker(ticker_symbol)
        expirations = tk.options
        if not expirations:
            return 0.0
        opt = tk.option_chain(expirations[0])
        calls = opt.calls
        if 'impliedVolatility' in calls.columns and not calls.empty:
            valid_iv = calls[calls['impliedVolatility'] > 0.01]['impliedVolatility']
            if not valid_iv.empty:
                return float(valid_iv.mean())
        return 0.0
    except Exception as e:
        print(f"Failed to fetch IV for {ticker_symbol}: {e}")
        return 0.0

def remove_secular_trend(prices: pd.Series, method: str = "linear") -> pd.Series:
    """
    Remove the secular trend from an asset's price series.
    Returns the detrended series (which represents cyclical/idiosyncratic movements).
    
    Methods:
    - linear: Simple linear regression detrending.
    - hp_filter: Hodrick-Prescott filter, decomposing into trend and cycle.
    """
    if prices.empty or len(prices) < 2:
        return prices

    prices_log = np.log(prices.dropna())

    if method == "linear":
        X = np.arange(len(prices_log))
        y = prices_log.values
        slope, intercept = np.polyfit(X, y, 1)
        trend = slope * X + intercept
        detrended_log = y - trend
        return pd.Series(detrended_log, index=prices_log.index)
        
    elif method == "hp_filter":
        # Lambda = 14400 / 129600 for daily data (or higher depending on literature)
        # We'll use lambda = 129600 for daily data
        cycle, trend = sm.tsa.filters.hpfilter(prices_log, 129600)
        return cycle

    return prices_log


def calculate_value_score(fundamentals: dict) -> float:
    """
    Calculate a composite value score based on PE, PB, and EV/EBITDA.
    Lower multiples mean a higher value score (better value).
    """
    score = 0
    valid_metrics = 0
    
    # Simple standardized scoring mechanism (lower is better, so we invert logic to make higher score = better value)
    
    pe = fundamentals.get("pe_ratio")
    if pe is not None and pe > 0:
        # P/E of 15 is neutral. If PE is 5, score = +10. If 25, score = -10.
        score += (15 - pe) * 0.1
        valid_metrics += 1
        
    pb = fundamentals.get("price_to_book")
    if pb is not None and pb > 0:
        # P/B of 2 is neutral. If PB is 1, score = +1. If 3, score = -1
        score += (2 - pb) * 1.0
        valid_metrics += 1
        
    ev = fundamentals.get("ev_to_ebitda")
    if ev is not None and ev > 0:
        # EV/EBITDA of 10 is neutral. 
        score += (10 - ev) * 0.2
        valid_metrics += 1
        
    if valid_metrics == 0:
        return 0.0
        
    return score / valid_metrics

def detrend_and_zscore(prices: pd.Series, window: int = 252) -> pd.Series:
    """
    Removes secular trend using HP filter and computes a rolling Z-score
    of the cyclical component to identify overbought/oversold conditions.
    """
    cycle = remove_secular_trend(prices, method="hp_filter")
    rolling_mean = cycle.rolling(window=window, min_periods=1).mean()
    rolling_std = cycle.rolling(window=window, min_periods=1).std()
    # avoid division by zero
    rolling_std = rolling_std.replace(0, pd.NA).bfill()
    z_score = (cycle - rolling_mean) / rolling_std
    return z_score
    
def generate_rationale(ticker: str, value_score: float, z_score: float, analyst_buy: int, funds: dict) -> str:
    """
    Generate a short text rationale.
    """
    rationale = []
    
    pe = funds.get("pe_ratio")
    pb = funds.get("price_to_book")
    
    if value_score > 0.5:
        base = "Undervalued."
        if pe and pe < 15: base += f" Low P/E ({pe:.1f}x)."
        if pb and pb < 2: base += f" Low P/B ({pb:.2f}x)."
        rationale.append(base.strip())
    elif value_score > 0:
        rationale.append("Fair valuation.")
    else:
        rationale.append("Expensive valuation.")
        
    if pd.isna(z_score):
        pass
    elif z_score < -1.5:
        rationale.append("Technically oversold.")
    elif z_score > 1.5:
        rationale.append("Strong upward momentum.")
    elif z_score > 0:
        rationale.append("Above historical trend.")
    else:
        rationale.append("Below historical trend.")
        
    if analyst_buy > 5:
        rationale.append("Strong Wall St consensus.")
    elif analyst_buy > 0:
        rationale.append("Positive analyst ratings.")
        
    if not rationale:
        return "Insufficient data."
        
    return " | ".join(rationale)

def calculate_probabilities(
    prices: pd.Series, 
    horizon_days: int = 14, 
    target_upside: float = 0.05, 
    max_downside: float = -0.05,
    n_paths: int = 1000,
    method: str = "garch"
) -> dict:
    if prices.empty or len(prices) < 252:
        return {"prob_success": 0.0, "prob_stop": 0.0, "exp_return": 0.0}
        
    try:
        if method == "historical":
            n_days = len(prices)
            if n_days < horizon_days + 1:
                return {"prob_success": 0.0, "prob_stop": 0.0, "exp_return": 0.0}
                
            wins = 0
            losses = 0
            total_trials = n_days - horizon_days
            
            for i in range(total_trials):
                path = prices.iloc[i:i+horizon_days+1]
                path_ret = path.values / path.values[0] - 1.0
                
                success_hits = (path_ret >= target_upside)
                stop_hits = (path_ret <= max_downside)
                
                success_met = np.any(success_hits)
                stop_met = np.any(stop_hits)
                
                if success_met and not stop_met:
                    wins += 1
                elif stop_met and not success_met:
                    losses += 1
                elif success_met and stop_met:
                    first_success = np.argmax(success_hits)
                    first_stop = np.argmax(stop_hits)
                    if first_success < first_stop:
                        wins += 1
                    else:
                        losses += 1
                        
            return {
                "prob_success": float(wins / total_trials),
                "prob_stop": float(losses / total_trials),
                "exp_return": float(prices.iloc[-1] / prices.iloc[-(horizon_days+1)] - 1.0)
            }
            
        elif method == "gbm":
            returns = prices.pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std()
            
            rng = np.random.default_rng(42)
            eps = rng.normal(size=(n_paths, horizon_days))
            
            r_sim = mu + sigma * eps
            r_sim = r_sim.astype(float).T
            
            cum_ret = np.cumprod(1.0 + r_sim, axis=0) - 1.0
            
            success_hits = (cum_ret >= target_upside)
            stop_hits = (cum_ret <= max_downside)
            
            success_met = np.any(success_hits, axis=0)
            stop_met = np.any(stop_hits, axis=0)
            
            first_success = np.argmax(success_hits, axis=0)
            first_stop = np.argmax(stop_hits, axis=0)
            
            wins = success_met & (~stop_met | (first_success < first_stop))
            losses = stop_met & (~success_met | (first_stop <= first_success))
            
            return {
                "prob_success": float(wins.mean()),
                "prob_stop": float(losses.mean()),
                "exp_return": float(cum_ret[-1, :].mean())
            }
            
        elif method in ["garch", "gjr-garch", "egarch", "aparch", "tgarch"]:
            returns_pct = 100 * prices.pct_change().dropna()
            if method == "gjr-garch":
                am = arch_model(returns_pct, mean="ARX", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t")
            elif method == "egarch":
                am = arch_model(returns_pct, mean="ARX", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t")
            elif method == "aparch":
                am = arch_model(returns_pct, mean="ARX", lags=1, vol="APARCH", p=1, o=1, q=1, dist="t")
            elif method == "tgarch":
                am = arch_model(returns_pct, mean="ARX", lags=1, vol="GARCH", p=1, o=1, q=1, power=1.0, dist="t")
            else: # garch
                am = arch_model(returns_pct, mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist="t")
            res = am.fit(disp="off")
            
            forecasts = res.forecast(horizon=horizon_days, method="simulation", simulations=n_paths)
            r_sim_pct = forecasts.simulations.values[-1, :, :] # shape: (n_paths, horizon_days)
            r_sim = (r_sim_pct / 100.0).astype(float).T
            
            cum_ret = np.cumprod(1.0 + r_sim, axis=0) - 1.0 # shape: (horizon_days, n_paths)
            
            success_hits = (cum_ret >= target_upside)
            stop_hits = (cum_ret <= max_downside)
            
            success_met = np.any(success_hits, axis=0)
            stop_met = np.any(stop_hits, axis=0)
            
            first_success = np.argmax(success_hits, axis=0)
            first_stop = np.argmax(stop_hits, axis=0)
            
            wins = success_met & (~stop_met | (first_success < first_stop))
            losses = stop_met & (~success_met | (first_stop <= first_success))
            
            prob_win = wins.mean()
            prob_loss = losses.mean()
            exp_return = cum_ret[-1, :].mean()
            
            return {
                "prob_success": float(prob_win),
                "prob_stop": float(prob_loss),
                "exp_return": float(exp_return)
            }
        else:
            return {"prob_success": 0.0, "prob_stop": 0.0, "exp_return": 0.0}
    except Exception as e:
        print(f"Prob calculation failed for {method}: {e}")
        return {"prob_success": 0.0, "prob_stop": 0.0, "exp_return": 0.0}
