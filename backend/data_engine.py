import yfinance as yf
import pandas as pd
import numpy as np
import ssl
from typing import List
from backend.database import get_universe, save_universe, get_fundamental_cache, save_fundamental_cache, save_analyst_rating, get_latest_analyst_rating

# Fix for macOS SSL certificate issues with pandas read_html
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


import requests
import io

def get_tickers_from_stockanalysis_holdings(url: str) -> List[str]:
    """
    Scrape ETF holdings to get universe of tickers.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        tables = pd.read_html(io.StringIO(res.text))
        if not tables:
            return []
        
        df = tables[0]
        symbol_col_candidates = [
            c for c in df.columns
            if "symbol" in str(c).lower() or "ticker" in str(c).lower()
        ]
        if not symbol_col_candidates:
            return []
            
        symbol_col = symbol_col_candidates[0]
        
        tickers = (
            df[symbol_col]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        
        # Clean up tickers for yfinance
        cleaned_tickers = []
        for t in tickers:
            if t.lower() == "nan" or t == "-" or ":" in t:
                continue
            cleaned_tickers.append(t.replace(".", "-"))
            
        return cleaned_tickers
    except Exception as e:
        print(f"Error fetching from {url}: {e}")
        return []

# Low-fee broad-market / style ETFs (Vanguard, iShares) and defense/aerospace names that
# never appear via the S&P500/Dow/Nasdaq100/Russell1000 index scrapes below.
SUPPLEMENTAL_TICKERS = [
    # Broad market
    "VTI", "VOO", "SPLG", "ITOT", "IVV",
    # Vanguard style/factor
    "VUG", "VTV", "VIG", "VYM", "VB", "VXUS",
    # iShares style/factor
    "IWM", "IWF", "IWD",
    # Defense & Aerospace ETFs
    "ITA", "PPA", "XAR", "SHLD",
    # Defense & Aerospace primes
    "LMT", "RTX", "NOC", "GD", "LHX", "HII", "TXT", "KTOS", "AVAV",
]

# Sector overrides applied ahead of yfinance's own `.info["sector"]` in get_fundamental_metrics().
# ETFs never get a sector from yfinance at all; the defense primes do (usually "Industrials"),
# so this has to win over yfinance rather than just filling a gap.
MANUAL_SECTOR_OVERRIDES = {
    ticker: "ETF" for ticker in [
        "VTI", "VOO", "SPLG", "ITOT", "IVV",
        "VUG", "VTV", "VIG", "VYM", "VB", "VXUS",
        "IWM", "IWF", "IWD",
    ]
}
MANUAL_SECTOR_OVERRIDES.update({
    ticker: "Defense & Aerospace" for ticker in [
        "ITA", "PPA", "XAR", "SHLD",
        "LMT", "RTX", "NOC", "GD", "LHX", "HII", "TXT", "KTOS", "AVAV",
    ]
})


def _merge_supplemental(tickers: List[str]) -> List[str]:
    """Dedupe-preserving-order union of `tickers` with SUPPLEMENTAL_TICKERS."""
    seen = set(tickers)
    merged = list(tickers)
    for t in SUPPLEMENTAL_TICKERS:
        if t not in seen:
            seen.add(t)
            merged.append(t)
    return merged


def build_universe(max_size: int = 2500, force_scrape: bool = False) -> List[str]:
    """
    Build stock universe by scraping major indices from Wikipedia.
    Includes S&P 500, Dow Jones, Nasdaq 100, and Russell 1000.
    Caches to history/universe.json.
    """
    import os
    import json
    import io
    import requests

    if not force_scrape:
        cached = get_universe()
        if cached and len(cached) > 3:
            merged = _merge_supplemental(cached)
            if len(merged) != len(cached):
                try:
                    save_universe(merged)
                except Exception as e:
                    print(f"Failed to save merged universe cache: {e}")
            return merged[:max_size]

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    def fetch_wiki_table(url):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            tables = pd.read_html(io.StringIO(res.text))
            for t in tables:
                if 'Symbol' in t.columns:
                    return t['Symbol'].tolist()
                elif 'Ticker' in t.columns:
                    return t['Ticker'].tolist()
            return []
        except Exception as e:
            print(f"Error fetching from {url}: {e}")
            return []

    all_tickers = []
    all_tickers.extend(fetch_wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"))
    all_tickers.extend(fetch_wiki_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"))
    all_tickers.extend(fetch_wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100"))
    all_tickers.extend(fetch_wiki_table("https://en.wikipedia.org/wiki/Russell_1000_Index"))

    # Clean up and deduplicate
    cleaned = []
    seen = set()
    for t in all_tickers:
        if pd.isna(t):
            continue
        t_str = str(t).strip().replace('.', '-')
        if t_str and t_str not in seen:
            seen.add(t_str)
            cleaned.append(t_str)
            
    if not cleaned:
        # Ultimate fallback - do NOT cache this!
        return ["AAPL", "MSFT", "GOOGL"][:max_size]

    cleaned = _merge_supplemental(cleaned)

    try:
        save_universe(cleaned)
    except Exception as e:
        print(f"Failed to save universe cache: {e}")
        
    if len(cleaned) > max_size:
        cleaned = cleaned[:max_size]
        
    return cleaned

def fetch_data(ticker: str, start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch daily historical data and calculate daily returns.
    Caches data to Parquet for fast historical access.
    """
    import os
    from datetime import datetime
    
    cache_dir = "backend/history/parquet_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}.parquet")
    
    if os.path.exists(cache_path):
        try:
            data = pd.read_parquet(cache_path)
            # Check if the cache is reasonably fresh (e.g., within 24 hours)
            if not data.empty:
                last_date = data.index[-1]
                if last_date.tz is not None:
                    last_date = last_date.tz_localize(None)
                if pd.Timestamp.now().normalize() - last_date.normalize() <= pd.Timedelta(days=1):
                    return data
        except Exception as e:
            print(f"Failed to load parquet cache for {ticker}: {e}")
            
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return data
        
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    data['Log_Ret'] = data['Close'].apply(lambda x: float(str(x)) if x is not None and not pd.isna(x) else float('nan')).apply(np.log).diff()
    
    try:
        data.to_parquet(cache_path)
    except Exception as e:
        print(f"Failed to save parquet cache for {ticker}: {e}")
        
    return data

import time
import os
import json
from datetime import datetime, timedelta

def _get_cached_yfinance(ticker: str, data_type: str, fetch_func, ttl_hours: int = 24):
    cached = get_fundamental_cache(ticker, data_type, ttl_hours)
    if cached is not None:
        return cached

    retries = 3
    for i in range(retries):
        try:
            data = fetch_func()
            save_fundamental_cache(ticker, data_type, data)
            return data
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)
            else:
                raise e

def fetch_analyst_ratings(ticker: str) -> dict:
    """
    Fetch analyst recommendations/ratings using yfinance.
    """
    def _fetch():
        tk = yf.Ticker(ticker)
        recs = tk.recommendations
        if recs is not None and not recs.empty:
            latest = recs.iloc[-1]
            return {
                "strongBuy": int(latest.get("strongBuy", 0)),
                "buy": int(latest.get("buy", 0)),
                "hold": int(latest.get("hold", 0)),
                "sell": int(latest.get("sell", 0)),
                "strongSell": int(latest.get("strongSell", 0)),
                "date": str(latest.get("period", "Unknown"))
            }
        return {"error": "No ratings found"}

    try:
        def _fetch_and_save():
            data = _fetch()
            if "error" not in data:
                save_analyst_rating(ticker, data)
            return data
            
        return _get_cached_yfinance(ticker, "ratings", _fetch_and_save, ttl_hours=24)
    except Exception as e:
        return {"error": "No ratings found"}

_sector_index = None
def load_sector_index():
    global _sector_index
    if _sector_index is None:
        try:
            import os
            import json
            idx_path = os.path.join(os.path.dirname(__file__), "sector_index.json")
            if os.path.exists(idx_path):
                with open(idx_path, 'r') as f:
                    _sector_index = json.load(f)
            else:
                _sector_index = {}
        except:
            _sector_index = {}
    return _sector_index

def _normalize_dividend_yield(value):
    if value is None:
        return 0.0
    try:
        yield_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if yield_value > 1:
        return yield_value / 100.0
    return yield_value


def get_fundamental_metrics(ticker: str) -> dict:
    """
    Get value-focused metrics from yfinance.
    """
    idx = load_sector_index()
    
    def _fetch():
        tk = yf.Ticker(ticker)
        info = tk.info
        
        sec = MANUAL_SECTOR_OVERRIDES.get(ticker) or info.get("sector")
        if not sec and ticker in idx and idx[ticker] != "Unknown Sector":
            sec = idx[ticker]
        elif not sec:
            sec = "Unknown Sector"

        raw_dividend_yield = info.get("dividendYield")
        if raw_dividend_yield is None:
            raw_dividend_yield = info.get("yield")
        dividend_yield = _normalize_dividend_yield(raw_dividend_yield)
        
        def _clamp(val, min_val=0.0):
            if val is None:
                return None
            try:
                val = float(val)
                return val if val >= min_val else None
            except (ValueError, TypeError):
                return None

        return {
            "pe_ratio": _clamp(info.get("trailingPE") or info.get("navPrice")),
            "forward_pe": _clamp(info.get("forwardPE")),
            "price_to_book": _clamp(info.get("priceToBook")),
            "ev_to_ebitda": _clamp(info.get("enterpriseToEbitda")),
            "market_cap": _clamp(info.get("marketCap") or info.get("totalAssets")),
            "dividend_yield": _clamp(dividend_yield),
            "beta": info.get("beta") or info.get("beta3Year") or 1.0,
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh") or info.get("previousClose") or None,
            "fifty_two_week_low": info.get("fiftyTwoWeekLow") or info.get("previousClose") or None,
            "sector": sec
        }
        
    try:
        return _get_cached_yfinance(ticker, "fundamentals", _fetch, ttl_hours=24)
    except:
        sec = idx.get(ticker, "Unknown Sector")
        return {"sector": sec}
