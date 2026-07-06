import os
import json
import time
import random
from datetime import datetime
from backend.data_engine import fetch_data, build_universe
from backend.metrics import calculate_rv, calculate_atr, calculate_hv_rank, fetch_current_iv

CACHE_FILE = os.path.join(os.path.dirname(__file__), "history", "volatility_metrics.json")

def update_metrics():
    print(f"[{datetime.now()}] Starting volatility metrics update...")
    
    # Ensure history directory exists
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    
    universe = build_universe(max_size=2500)
    print(f"Loaded {len(universe)} tickers from universe.")
    
    # Load existing cache to avoid losing data if script is interrupted
    metrics_cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                metrics_cache = json.load(f)
        except:
            pass

    for idx, tk in enumerate(universe):
        print(f"[{idx+1}/{len(universe)}] Processing {tk}...")
        try:
            # We need ~1.5 years of data (approx 380 days) for 252-day lookback + 21-day rolling window
            # 2 years is safe
            start_date = (datetime.now() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
            data = fetch_data(tk, start=start_date)
            
            rv = 0.0
            atr = 0.0
            hv_rank = 0.0
            current_iv = 0.0
            
            if not data.empty:
                rv = calculate_rv(data['Close'])
                if 'High' in data.columns and 'Low' in data.columns:
                    atr = calculate_atr(data['High'], data['Low'], data['Close'])
                hv_rank = calculate_hv_rank(data['Close'])
                
            current_iv = fetch_current_iv(tk)
            
            metrics_cache[tk] = {
                "rv": rv,
                "atr": atr,
                "hv_rank": hv_rank,
                "current_iv": current_iv,
                "last_updated": datetime.now().isoformat()
            }
            
            # Save incrementally every 10 tickers
            if (idx + 1) % 10 == 0:
                with open(CACHE_FILE, "w") as f:
                    json.dump(metrics_cache, f, indent=4)
                    
        except Exception as e:
            print(f"Failed to process {tk}: {e}")
            
        # Respect rate limits
        time.sleep(random.uniform(0.5, 1.5))
        
    # Final save
    with open(CACHE_FILE, "w") as f:
        json.dump(metrics_cache, f, indent=4)
        
    print(f"[{datetime.now()}] Finished volatility metrics update.")

if __name__ == "__main__":
    import pandas as pd
    update_metrics()
