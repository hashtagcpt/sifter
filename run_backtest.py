import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
from backend.metrics import detrend_and_zscore, calculate_biweekly_probabilities
from backend.data_engine import build_universe
import concurrent.futures

def run_historical_backtest():
    print("Starting Weekly Backtest over the last 1 year on the full universe...")
    
    print("Fetching expanded ETF universe...")
    universe = build_universe(max_size=250)
    print(f"Universe size: {len(universe)}")
    
    print("Downloading historical data (this may take a minute)...")
    data = yf.download(universe, period="3y", auto_adjust=True, progress=False)['Close']
    
    total_days = len(data)
    if total_days < 252:
        print("Not enough data to backtest 1 year.")
        return
        
    backtest_start_idx = total_days - 252
    
    capital = 1000.0
    equity_curve = [capital]
    dates = []
    
    print("Running weekly simulations...")
    
    for i in range(backtest_start_idx, total_days - 5, 5):
        current_date = data.index[i]
        dates.append(current_date)
        
        historical_slice = data.iloc[:i]
        
        # Calculate 60-day correlation matrix
        recent_returns = historical_slice.iloc[-60:].pct_change().dropna()
        corr_matrix = recent_returns.corr()
        
        weekly_scores = []
        
        # We can use ProcessPoolExecutor or ThreadPoolExecutor to speed up GARCH fits slightly
        def process_ticker(tk):
            if tk not in historical_slice.columns: return None
            tk_prices = historical_slice[tk].dropna()
            if len(tk_prices) < 252:
                return None
                
            z = detrend_and_zscore(tk_prices, window=126)
            z_score = z.iloc[-1] if not pd.isna(z.iloc[-1]) else 0.0
            
            probs = calculate_biweekly_probabilities(tk_prices, horizon_days=14, target_upside=0.05, max_downside=-0.05, n_paths=200) # reduced paths for speed
            
            # Anti-momentum / Mean-reversion
            rank_score = (probs["prob_success"] * 10.0) - (float(z_score) * 1.0)
            
            return {
                "ticker": tk,
                "rank_score": rank_score,
                "prob_success": probs["prob_success"],
                "z_score": z_score
            }

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_ticker, universe)
            for res in results:
                if res is not None:
                    weekly_scores.append(res)
            
        weekly_scores.sort(key=lambda x: x["rank_score"], reverse=True)
        if not weekly_scores:
            equity_curve.append(capital)
            continue
            
        # Select Top 3 uncorrelated assets
        selected_portfolio = []
        for item in weekly_scores:
            tk = item["ticker"]
            if len(selected_portfolio) >= 3:
                break
                
            # Check correlation against already selected
            is_uncorrelated = True
            for sel_tk in selected_portfolio:
                if tk in corr_matrix.columns and sel_tk in corr_matrix.columns:
                    correlation = corr_matrix.loc[tk, sel_tk]
                    if correlation > 0.4: # strict correlation threshold
                        is_uncorrelated = False
                        break
            
            if is_uncorrelated:
                selected_portfolio.append(tk)
                
        # If we couldn't find 3 uncorrelated, just take the absolute top ones to fill it out
        if len(selected_portfolio) < 3:
            for item in weekly_scores:
                if item["ticker"] not in selected_portfolio:
                    selected_portfolio.append(item["ticker"])
                if len(selected_portfolio) >= 3:
                    break
                    
        # "Buy" top 3 stocks, allocate evenly
        portfolio_return = 0.0
        trade_logs = []
        
        for tk in selected_portfolio:
            entry_price = data[tk].iloc[i]
            exit_price = data[tk].iloc[i + 5]
            tk_ret = (exit_price / entry_price) - 1.0
            portfolio_return += (tk_ret / 3.0)
            trade_logs.append(f"{tk} (Price: ${entry_price:.2f})")
            
        # Update capital
        trade_pnl = capital * portfolio_return
        capital += trade_pnl
        equity_curve.append(capital)
        
        print(f"{current_date.date()} | Portfolio: {', '.join(trade_logs)} | 1-Wk Port Ret: {portfolio_return*100:5.2f}% | Capital: ${capital:.2f}")

    total_return = (capital / 1000.0) - 1.0
    equity_series = pd.Series(equity_curve)
    max_drawdown = (equity_series / equity_series.cummax() - 1).min()
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS (1 Year)")
    print("="*40)
    print(f"Starting Capital : $1000.00")
    print(f"Ending Capital   : ${capital:.2f}")
    print(f"Total Return     : {total_return * 100:.2f}%")
    print(f"Max Drawdown     : {max_drawdown * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_historical_backtest()
