import numpy as np
import pandas as pd
import yfinance as yf
from backend.data_engine import build_universe

def compute_return_metrics(strategy_return: pd.Series) -> dict:
    """
    Compute the standard return/risk metrics shared by all backtest-style functions
    in this module, from a per-period strategy_return series (buy-and-hold or
    signal-driven - this function doesn't care which).

    Returns {"metrics": {...}, "cum_return": pd.Series, "drawdown": pd.Series}.
    "metrics" does NOT include trades_count - callers that have a position/trade
    series (single-asset and pairs backtests) splice that in themselves.
    """
    cum_return = (1 + strategy_return).cumprod()

    total_return = cum_return.iloc[-1] - 1 if not cum_return.empty else 0
    annualized_return = (1 + total_return) ** (252 / len(strategy_return)) - 1 if len(strategy_return) > 0 else 0

    daily_vol = strategy_return.std()
    annualized_vol = daily_vol * np.sqrt(252)

    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Sortino (downside deviation)
    downside_returns = strategy_return[strategy_return < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = annualized_return / downside_vol if downside_vol > 0 else 0

    # Maximum Drawdown
    roll_max = cum_return.cummax()
    drawdown = cum_return / roll_max - 1.0
    max_drawdown = drawdown.min()

    # Win rate
    wins = (strategy_return > 0).sum()
    losses = (strategy_return < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    return {
        "metrics": {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_vol": float(annualized_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
        },
        "cum_return": cum_return,
        "drawdown": drawdown,
    }


def run_vectorized_backtest(
    prices: pd.Series,
    signals: pd.Series,
    transaction_cost: float = 0.001,
    borrow_cost_annual: float = 0.03
) -> dict:
    """
    Run a vectorized backtest for a single asset.
    signals: pd.Series of target positions (-1 for short, 1 for long, 0 for flat).
    """
    if prices.empty or signals.empty:
        return {"error": "Empty prices or signals"}

    df = pd.DataFrame({"price": prices, "signal": signals}).dropna()
    df['return'] = df['price'].pct_change().fillna(0)

    # We enter the trade on the day the signal is generated, so return is realized the next day
    # Or, typical convention: signal at t means position at t to capture return t+1.
    df['position'] = df['signal'].shift(1).fillna(0)

    # Daily borrow cost for short positions
    daily_borrow = borrow_cost_annual / 252
    df['borrow_fee'] = np.where(df['position'] < 0, df['position'].abs() * daily_borrow, 0)

    # Transaction costs applied when position changes
    df['trade'] = df['position'].diff().fillna(0)
    df['tc'] = df['trade'].abs() * transaction_cost

    df['strategy_return'] = df['position'] * df['return'] - df['tc'] - df['borrow_fee']

    result = compute_return_metrics(df['strategy_return'])
    result["metrics"]["trades_count"] = float(df['trade'].abs().sum())

    return {
        "metrics": result["metrics"],
        "timeseries": {
            "dates": df.index.astype(str).tolist(),
            "cum_return": result["cum_return"].tolist(),
            "drawdown": result["drawdown"].tolist()
        }
    }

def run_vectorized_pairs_backtest(
    prices_y: pd.Series,
    prices_x: pd.Series,
    signals: pd.Series,
    transaction_cost: float = 0.002,
    borrow_cost_annual: float = 0.03
) -> dict:
    """
    Run a vectorized backtest for a statistical arbitrage pair.
    """
    if prices_y.empty or prices_x.empty or signals.empty:
        return {"error": "Empty prices or signals"}
        
    df = pd.DataFrame({"price_y": prices_y, "price_x": prices_x, "signal": signals}).dropna()
    df['ret_y'] = df['price_y'].pct_change().fillna(0)
    df['ret_x'] = df['price_x'].pct_change().fillna(0)
    
    df['position'] = df['signal'].shift(1).fillna(0)
    
    daily_borrow = borrow_cost_annual / 252
    df['borrow_fee'] = np.where(df['position'] != 0, daily_borrow, 0)
    
    df['trade'] = df['position'].diff().fillna(0)
    df['tc'] = df['trade'].abs() * transaction_cost * 2
    
    df['strategy_return'] = df['position'] * 0.5 * (df['ret_y'] - df['ret_x']) - df['tc'] - df['borrow_fee']

    result = compute_return_metrics(df['strategy_return'])
    result["metrics"]["trades_count"] = float(df['trade'].abs().sum())

    return {
        "metrics": result["metrics"],
        "timeseries": {
            "dates": df.index.astype(str).tolist(),
            "cum_return": result["cum_return"].tolist(),
            "drawdown": result["drawdown"].tolist()
        }
    }

def find_top_sortino_baskets(
    basket_size: int = 5,
    num_baskets: int = 5,
    sample_size: int = 100,
    iterations: int = 10000
) -> list:
    import random
    
    universe = build_universe(max_size=sample_size)
    if not universe:
        return []
        
    data = yf.download(universe, period="1mo", progress=False)
    if data.empty or 'Close' not in data.columns:
        return []
        
    prices = data['Close']
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        
    returns = prices.pct_change().fillna(0)
    returns = returns.dropna(axis=1, thresh=len(returns) - 5)
    available_tickers = list(returns.columns)
    
    baskets = []
    
    for b in range(num_baskets):
        if len(available_tickers) < basket_size:
            break
            
        best_sortino = -np.inf
        best_basket = []
        best_metrics = {}
        
        for _ in range(iterations):
            sample = random.sample(available_tickers, basket_size)
            port_returns = returns[sample].mean(axis=1)
            
            cum_ret = (1 + port_returns).cumprod()
            tot_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
            ann_ret = (1 + tot_ret) ** (252 / len(port_returns)) - 1 if len(port_returns) > 0 else 0
            
            downside = port_returns[port_returns < 0]
            downside_vol = downside.std() * np.sqrt(252)
            
            if downside_vol > 0:
                sortino = ann_ret / downside_vol
            else:
                sortino = 0
                
            if sortino > best_sortino:
                best_sortino = sortino
                best_basket = sample
                best_metrics = {
                    "sortino_ratio": float(sortino),
                    "total_return": float(tot_ret),
                    "annualized_return": float(ann_ret)
                }
                
        if not best_basket:
            break
            
        baskets.append({
            "tickers": best_basket,
            "metrics": best_metrics
        })
        
        for t in best_basket:
            available_tickers.remove(t)
            
    return baskets
