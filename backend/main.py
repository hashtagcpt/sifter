from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
from typing import Dict
import numpy as np

from backend.data_engine import build_universe, fetch_data, get_fundamental_metrics, fetch_analyst_ratings
from backend.metrics import calculate_value_score, detrend_and_zscore
from backend.strategy_bayesian import apply_bayesian_reweighting
from backend.stat_arb import calculate_cointegration, estimate_half_life, generate_z_scores
from backend.factor_models import extract_pca_factors, rolling_factor_regression
from backend.backtester import run_vectorized_backtest, run_vectorized_pairs_backtest, find_top_sortino_baskets
from backend.database import (
    init_db, get_universe, save_universe, 
    save_bayesian_view, get_bayesian_views, 
    save_ai_macro_scores, get_ai_macro_score_history, 
    get_analyst_rating_history
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import subprocess
import atexit

import shutil

ollama_process = None
try:
    ollama_path = shutil.which("ollama") or "/usr/local/bin/ollama"
    ollama_process = subprocess.Popen([ollama_path, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception as e:
    print(f"Failed to auto-start ollama: {e}")

def cleanup_ollama():
    if ollama_process:
        try:
            ollama_process.terminate()
        except:
            pass
atexit.register(cleanup_ollama)

app = FastAPI(title="Sifter")

@app.on_event("startup")
def check_ollama_status():
    import time
    import requests
    import sys
    
    print("Checking Ollama status...")
    max_retries = 5
    for i in range(max_retries):
        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=5)
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name") for m in data.get("models", [])]
                required_models = ["gemma4:e4b", "llama3.1:8b", "gemma4:latest"]
                available = [rm for rm in required_models if rm in models]
                
                if not available:
                    print(f"CRITICAL ERROR: None of the required models are available. Checked: {required_models}")
                    sys.exit(1)
                else:
                    if "gemma4:e4b" not in available:
                        print(f"WARNING: Preferred model 'gemma4:e4b' is missing, but fallbacks are available: {available}")
                    else:
                        print(f"Ollama is running. Supported models found: {available}")
                    return
        except requests.exceptions.RequestException:
            pass
        
        print(f"Waiting for Ollama to start... ({i+1}/{max_retries})")
        time.sleep(2)
        
    print("CRITICAL ERROR: Ollama failed to start or is not responding.")
    sys.exit(1)


history_dir = os.path.join(os.path.dirname(__file__), "history")
os.makedirs(history_dir, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional static files for the frontend so we don't need npm
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize SQLite database
init_db()

SECTOR_SENTIMENT_PLAYBOOK = {
    "Technology": "product cycles, enterprise/cloud demand, AI adoption that converts to revenue, margin durability, capex intensity, and competitive displacement",
    "Financial Services": "net interest margins, credit quality, deposit flows, capital markets activity, regulation, loan growth, and yield-curve direction",
    "Healthcare": "clinical/regulatory milestones, reimbursement, patent cliffs, procedure volumes, utilization trends, and balance-sheet capacity",
    "Consumer Cyclical": "employment, real wage growth, consumer confidence, rates, discretionary spending, inventory discipline, and pricing power",
    "Industrials": "orders/backlog, infrastructure spending, aerospace/defense cycles, reshoring, input costs, and operating leverage",
    "Energy": "oil and gas supply-demand, OPEC policy, inventory levels, refining margins, discoveries, reserve quality, and capex discipline",
    "Utilities": "rate cases, allowed ROE, load growth, fuel costs, grid investment, weather normalization, and interest-rate sensitivity",
    "Real Estate": "occupancy, lease spreads, cap rates, refinancing risk, property-type demand, and interest-rate sensitivity",
    "Basic Materials": "commodity prices, China/global demand, mine supply, inventories, energy costs, and capacity additions or disruptions",
    "Communication Services": "advertising demand, subscriber trends, content costs, regulation, broadband/wireless competition, and platform engagement",
    "Consumer Defensive": "unit volumes, private-label pressure, commodity/input costs, pricing power, shelf space, and trade-down resilience",
    "Unknown Sector": "company-specific demand, balance-sheet strength, valuation, competitive position, management execution, and industry news",
}

SECTOR_SENTIMENT_CACHE_HOURS = 8


def _parse_json_object(text: str) -> dict:
    import json
    import re

    if not text:
        return {}
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    clean_text = json_match.group(1) if json_match else text.strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        object_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if object_match:
            return json.loads(object_match.group(0))
    return {}


def _sentiment_cache_path(ticker: str, sector: str, model: str) -> str:
    import re

    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{ticker}_{sector}_{model}")[:120]
    return os.path.join(history_dir, "sentiment_cache", f"{safe_key}.json")


def _load_cached_sector_note(ticker: str, sector: str, model: str):
    import json
    from datetime import timedelta

    path = _sentiment_cache_path(ticker, sector, model)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            cached = json.load(f)
        cache_time = datetime.fromisoformat(cached.get("timestamp", ""))
        if datetime.now() - cache_time < timedelta(hours=SECTOR_SENTIMENT_CACHE_HOURS):
            return cached.get("data")
    except Exception:
        return None
    return None


def _save_cached_sector_note(ticker: str, sector: str, model: str, data: dict):
    import json

    path = _sentiment_cache_path(ticker, sector, model)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f)
    except Exception as e:
        print(f"Failed to save sentiment cache for {ticker}: {e}")


def build_ticker_sentiment_note(ticker: str, model: str = "gemma4:e4b") -> dict:
    """
    Use yfinance news and DDG peer news plus the local LLM to produce a one-sentence,
    ticker-aware sentiment note.
    """
    try:
        import requests
        import yfinance as yf
        from ddgs import DDGS

        # Check a generic cache so we don't spam Ollama too much
        cached = _load_cached_sector_note(ticker, "all", model)
        if cached:
            return cached

        # 1. Fetch exact ticker news
        tk_obj = yf.Ticker(ticker)
        yf_news = tk_obj.news
        headlines = []
        sources = []
        if yf_news:
            for item in yf_news[:5]:
                content = item.get("content", {})
                title = content.get("title", "") or item.get("title", "")
                publisher = content.get("provider", {}).get("displayName", "") or item.get("publisher", "")
                
                url_data = content.get("canonicalUrl")
                link = url_data.get("url", "") if isinstance(url_data, dict) else item.get("link", "")
                
                if title:
                    headlines.append(f"- {title} ({publisher})")
                    if link:
                        sources.append({"title": title, "url": link, "snippet": publisher})
            
        # 2. Search DDGS for competitors/recent news
        query = f'{ticker} stock competitors news'
        results = list(DDGS().text(query, max_results=3))
        peer_sources = []
        for r in results:
            if r.get("href"):
                peer_sources.append(f"- {r.get('title', '')} ({r.get('href', '')}): {r.get('body', '')}")
                sources.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
        
        if not sources:
            return {
                "sentence": "",
                "sources": [],
                "status": "no_sources",
            }
            
        combined_context = "Recent Company Headlines:\n" + "\n".join(headlines) + "\n\nCompetitor & Market Context:\n" + "\n".join(peer_sources)

        prompt = f"""
You are an expert equity analyst.
Ticker: {ticker}

Recent web context:
{combined_context}

Analyze the immediate short-term catalysts for this stock based on the news above.
Return ONLY valid JSON with:
{{
  "sentiment": "Bullish, Bearish, or Neutral",
  "rationale": "A punchy, 1-2 sentence rationale citing the specific news or competitor context."
}}
"""
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }, timeout=300)
        
        if res.status_code != 200:
            raise RuntimeError(f"Ollama API failed with status {res.status_code}")

        llm_data = _parse_json_object(res.json().get("response", "{}"))
        
        sentiment = str(llm_data.get("sentiment", "Neutral")).strip()
        rationale = str(llm_data.get("rationale", "")).strip()
        
        if rationale:
            sentence = f"**{sentiment}**: {rationale}"
        else:
            sentence = ""
            
        note = {
            "sentence": sentence,
            "sources": sources[:3], # Return top 3 sources to frontend
            "status": "success" if sentence else "empty",
        }
        _save_cached_sector_note(ticker, "all", model, note)
        return note
    except Exception as e:
        print(f"Sentiment note error for {ticker}: {e}")
        return {
            "sentence": "",
            "sources": [],
            "status": "error"
        }

class UniverseRequest(BaseModel):
    max_size: int = 2500
    
class UniverseAddRequest(BaseModel):
    ticker: str

class ViewsRequest(BaseModel):
    views: Dict[str, float]
    confidences: Dict[str, float]
    
@app.get("/api/health")
def read_root(): return {"status": "ok", "message": "Quant Engine Running"}

@app.post("/api/universe")
def generate_universe(req: UniverseRequest):
    uni = build_universe(req.max_size)
    if not uni: uni = ["AAPL", "MSFT", "GOOGL"] # fallback
    return {"universe": uni}

@app.post("/api/scrape_universe")
def scrape_universe(req: UniverseRequest):
    uni = build_universe(req.max_size, force_scrape=True)
    if not uni: uni = ["AAPL", "MSFT", "GOOGL"] # fallback
    return {"universe": uni, "count": len(uni)}

from backend.metrics import generate_rationale, calculate_probabilities

@app.get("/api/screener")
def get_screener_results(prob_model: str = "garch"):
    uni = get_universe()
    if not uni: uni = ["AAPL", "MSFT", "GOOGL"]
    results = []
    
    for tk in uni:
        try:
            # Fundamentals and Ratings
            funds = get_fundamental_metrics(tk)
            val_score = calculate_value_score(funds)
            ratings = fetch_analyst_ratings(tk)
            analyst_buy = ratings.get("buy", 0) + ratings.get("strongBuy", 0)
            
            # Historical momentum (detrended Z-score)
            data = fetch_data(tk, start="2020-01-01") # grab historical context
            z_score = pd.NA
            if not data.empty and len(data) > 130:
                z = detrend_and_zscore(data['Close'], window=126)
                z_score = z.iloc[-1] if not pd.isna(z.iloc[-1]) else 0.0
            else:
                z_score = 0.0
                
            from backend.metrics import calculate_probabilities
            biweekly_probs = calculate_probabilities(data['Close'], method=prob_model) if not data.empty else {"prob_success": 0.0, "prob_stop": 0.0, "exp_return": 0.0}
                
            rationale = generate_rationale(tk, val_score, float(z_score), analyst_buy, funds)
            
            # Raw composite score (Value + Momentum Reversal + Analyst + Probability)
            raw_rank_score = val_score - (float(z_score) * 1.0) + (analyst_buy * 0.05) + (biweekly_probs["prob_success"] * 10.0)

            
            results.append({
                "ticker": tk,
                "value_score": val_score,
                "z_score": float(z_score),
                "analyst_ratings": ratings,
                "fundamentals": funds,
                "rationale": rationale,
                "biweekly_probs": biweekly_probs,
                "raw_rank_score": raw_rank_score
            })
        except Exception as e:
            print(f"Error screening {tk}: {e}")
            
    # Calculate Percentile Rank (0 to 100)
    results.sort(key=lambda x: x["raw_rank_score"])
    n = len(results)
    for i, res in enumerate(results):
        res["rank_score"] = (i / max(1, n - 1)) * 100.0 if n > 0 else 50.0
            
    # Sort by rank_score descending
    results.sort(key=lambda x: x["rank_score"], reverse=True)
    
    return {"data": results}

@app.get("/api/analyze_ticker/{tk}")
def analyze_ticker(
    tk: str,
    upside: float = 0.05,
    downside: float = -0.05,
    horizon: int = 30,
    prob_model: str = "garch",
    news_notes: bool = True,
    model: str = "gemma4:e4b",
):
    try:
        # Fundamentals and Ratings
        funds = get_fundamental_metrics(tk)
        val_score = calculate_value_score(funds)
        ratings = fetch_analyst_ratings(tk)
        analyst_buy = ratings.get("buy", 0) + ratings.get("strongBuy", 0)
        
        # Historical momentum (detrended Z-score)
        data = fetch_data(tk, start="2020-01-01") # grab historical context
        z_score = pd.NA
        if not data.empty and len(data) > 130:
            z = detrend_and_zscore(data['Close'], window=126)
            z_score = z.iloc[-1] if not pd.isna(z.iloc[-1]) else 0.0
        else:
            z_score = 0.0
            
        from backend.metrics import calculate_probabilities
        biweekly_probs = calculate_probabilities(data['Close'], horizon_days=horizon, target_upside=upside, max_downside=downside, method=prob_model) if not data.empty else {"prob_success": 0.0, "prob_stop": 0.0, "exp_return": 0.0}
        
        def get_cached_volatility_metrics(tk_sym: str) -> dict:
            import os, json
            cache_file = os.path.join(os.path.dirname(__file__), "history", "volatility_metrics.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        cache = json.load(f)
                        if tk_sym in cache:
                            return cache[tk_sym]
                except Exception:
                    pass
            return None
            
        vol_metrics = get_cached_volatility_metrics(tk)
        if vol_metrics is None:
            from backend.metrics import calculate_rv, calculate_atr, calculate_hv_rank, fetch_current_iv
            rv = calculate_rv(data['Close']) if not data.empty else 0.0
            atr = calculate_atr(data['High'], data['Low'], data['Close']) if not data.empty and 'High' in data.columns and 'Low' in data.columns else 0.0
            hv_rank = calculate_hv_rank(data['Close']) if not data.empty else 0.0
            current_iv = fetch_current_iv(tk)
        else:
            rv = vol_metrics.get("rv", 0.0)
            atr = vol_metrics.get("atr", 0.0)
            hv_rank = vol_metrics.get("hv_rank", 0.0)
            current_iv = vol_metrics.get("current_iv", 0.0)
            
        current_price = float(data['Close'].iloc[-1]) if not data.empty else 0.0
            
        rationale = generate_rationale(tk, val_score, float(z_score), analyst_buy, funds)
        sentiment_note = None
        if news_notes:
            from backend.main import build_ticker_sentiment_note
            sentiment_note = build_ticker_sentiment_note(tk, model)
        

        # Raw composite score (Value + Momentum Reversal + Analyst + Probability)
        raw_rank_score = val_score - (float(z_score) * 1.0) + (analyst_buy * 0.05) + (biweekly_probs["prob_success"] * 10.0)
        
        # We can't do percentile rank for a single stock easily without loading the universe, 
        # so we return the raw score as rank_score, or a placeholder if frontend normalizes.
        # But we'll provide both.
        rank_score = raw_rank_score
        
        return {
            "status": "success",
            "data": {
                "ticker": tk,
                "current_price": current_price,
                "value_score": val_score,
                "z_score": float(z_score),
                "analyst_ratings": ratings,
                "fundamentals": funds,
                "rationale": rationale,
                "sentiment_note": sentiment_note,
                "biweekly_probs": biweekly_probs,
                "rv": rv,
                "atr": atr,
                "hv_rank": hv_rank,
                "current_iv": current_iv,
                "raw_rank_score": raw_rank_score,
                "rank_score": rank_score
            }
        }
    except Exception as e:
        print(f"Error screening {tk}: {e}")
        return {"status": "error", "message": str(e), "ticker": tk}

@app.post("/api/bayesian_update")
def apply_views(req: ViewsRequest):
    for tk, prob in req.views.items():
        conf = req.confidences.get(tk, 0.5)
        save_bayesian_view(tk, prob, conf)
        
    uni = get_universe()
    if not uni: uni = ["AAPL", "MSFT", "GOOGL"]
    base_weights = {tk: 1.0 / len(uni) for tk in uni}
    
    views, confidences = get_bayesian_views()
    new_weights = apply_bayesian_reweighting(base_weights, views, confidences)
    
    return {"new_weights": new_weights}

@app.get("/api/macro")
def get_macro_factors():
    try:
        import yfinance as yf
        # yf.Tickers is good for fetching info
        macro = yf.Tickers("^VIX ^TNX")
        vix_price = macro.tickers['^VIX'].info.get('regularMarketPrice') or macro.tickers['^VIX'].info.get('previousClose', 0.0)
        tnx_price = macro.tickers['^TNX'].info.get('regularMarketPrice') or macro.tickers['^TNX'].info.get('previousClose', 0.0)
        
        # Get SPY trend using existing logic
        spy_data = fetch_data("SPY", start="2020-01-01")
        spy_z = 0.0
        if not spy_data.empty and len(spy_data) > 130:
            z = detrend_and_zscore(spy_data['Close'], window=126)
            spy_z = float(z.iloc[-1]) if not pd.isna(z.iloc[-1]) else 0.0
            
        return {
            "vix": vix_price,
            "tnx": tnx_price,
            "spy_zscore": spy_z
        }
    except Exception as e:
        print(f"Error fetching macro: {e}")
        return {"vix": 0.0, "tnx": 0.0, "spy_zscore": 0.0, "error": str(e)}


@app.get("/api/ollama_models")
def get_ollama_models():
    try:
        import requests
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        if res.status_code == 200:
            models = res.json().get("models", [])
            return {"status": "success", "models": [m["name"] for m in models]}
    except Exception as e:
        print(f"Error fetching models: {e}")
    return {"status": "error", "models": []}

from datetime import datetime
from fastapi import Request, BackgroundTasks

class ReportRequest(BaseModel):
    filename: str
    prob_model: str
    model: str = "gemma4:e4b"

def generate_llm_report(req: ReportRequest):
    import json
    import os
    import requests
    
    path = os.path.join(history_dir, req.filename)
    if not os.path.exists(path):
        print(f"Report generation failed: {req.filename} not found.")
        return
        
    try:
        with open(path, "r") as f:
            run_data = json.load(f)
            
        analysis = run_data.get("analysis", [])
        if not analysis:
            return
            
        # Extract top 50 stocks to give the AI a broader view of the ranking metric
        top_stocks = analysis[:50]
        stock_list_str = "\n".join([
            f"- {s.get('ticker')}: Sector: {s.get('fundamentals', {}).get('sector', 'Unknown')} | "
            f"Rank Score: {s.get('rank_score', 0):.2f} | Value: {s.get('value_score', 0):.2f} | "
            f"Z-Score (Momentum): {s.get('z_score', 0):.2f} | Prob Success: {s.get('biweekly_probs', {}).get('prob_success', 0):.2f}" 
            for s in top_stocks
        ])
        macro_scores = run_data.get("macroScores", {})
        
        prompt = f'''
You are an expert, creative equity analyst and portfolio manager. A quantitative screener has just run using the '{req.prob_model}' model and generated metrics for the market.
Here are the top 50 stocks ranked by the screener, along with their key metrics:
{stock_list_str}

Here are the macro sector adjustments currently applied:
{json.dumps(macro_scores, indent=2)}

Your task:
1. Provide a global commentary on the ranking metrics you see across these top 50 stocks and the macro environment. What themes, sectors, or factors (value, momentum, probability) stand out to you right now?
2. From the list of 50 stocks above, creatively select your OWN Top 10 stocks that you believe offer the best opportunity. Do not just pick the top 10 by Rank Score. Combine the metrics creatively (e.g., finding a balance of value and momentum, or leaning into high probability setups).
3. Clearly list your 10 chosen stocks and provide a compelling, creative rationale for each pick based on its metrics and sector.

Format the output as clean HTML (without ```html markdown blocks). Use <h2> for headings, <ul> / <ol> for lists, and <p> for paragraphs.
Do NOT include <html> or <body> tags, just the inner HTML content.
'''
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": req.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 2048
            }
        }, timeout=900)
        
        if res.status_code == 200:
            llm_html = res.json().get("response", "<p>Error generating report.</p>")
            llm_html = llm_html.replace("```html", "").replace("```", "").strip()
            
            final_html = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Latest Model Report - {req.prob_model}</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 40px auto; padding: 20px; }}
                    h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 32px; }}
                    p {{ margin-bottom: 16px; }}
                    ul {{ background: #f8f9fa; padding: 20px 40px; border-radius: 8px; border: 1px solid #e9ecef; margin-bottom: 24px; }}
                    li {{ margin-bottom: 8px; }}
                </style>
            </head>
            <body>
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #2c3e50; margin-bottom: 8px;">Model Run Analysis</h1>
                    <div style="display: inline-block; background: #e8f4fd; color: #0d6efd; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9em;">
                        Model: {req.prob_model.upper()}
                    </div>
                </div>
                {llm_html}
            </body>
            </html>
            '''
            out_path = os.path.join(static_dir, "latest_model_report.html")
            with open(out_path, "w") as out_f:
                out_f.write(final_html)
    except Exception as e:
        print(f"Error generating LLM report: {e}")

@app.post("/api/generate_report")
def api_generate_report(req: ReportRequest):
    generate_llm_report(req)
    return {"status": "success", "message": "Report generation finished"}

@app.get("/api/history")
def list_history():
    files = [f for f in os.listdir(history_dir) if f.endswith(".json") and f.startswith("run_")]
    files.sort(reverse=True)
    
    history_data = []
    import json
    import yfinance as yf
    from datetime import datetime, timedelta
    
    needs_price = set()
    parsed_files = []
    
    for filename in files:
        path = os.path.join(history_dir, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            timestamp_str = data.get("timestamp", "")
            if not timestamp_str:
                history_data.append(filename)
                continue
                
            try:
                if timestamp_str.endswith('Z'):
                    timestamp = datetime.fromisoformat(timestamp_str[:-1])
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
            except:
                history_data.append(filename)
                continue
                
            prob_model = data.get("prob_model", "garch")
            horizon = data.get("horizon", 14)
            analysis = data.get("analysis", [])
            
            if not analysis:
                history_data.append(filename)
                continue
                
            top_3 = analysis[:3]
            top_3_tickers = [x["ticker"] for x in top_3]
            initial_cost = sum(x.get("current_price", 0.0) for x in top_3)
            
            is_expired = False
            realized_return = data.get("realized_return")
            
            if timestamp + timedelta(days=horizon) < datetime.utcnow():
                is_expired = True
                if realized_return is None:
                    needs_price.update(top_3_tickers)
            else:
                needs_price.update(top_3_tickers)
            
            parsed_files.append({
                "filename": filename,
                "data_ref": data,
                "path": path,
                "timestamp": timestamp_str,
                "prob_model": prob_model,
                "horizon": horizon,
                "top_3_tickers": top_3_tickers,
                "initial_cost": initial_cost,
                "is_expired": is_expired,
                "realized_return": realized_return
            })
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            history_data.append(filename)
            
    current_prices = {}
    if needs_price:
        try:
            tickers_list = list(needs_price)
            df = yf.download(tickers_list, period="1d")
            if not df.empty and 'Close' in df:
                close_df = df['Close']
                if len(tickers_list) == 1:
                    current_prices[tickers_list[0]] = float(close_df.iloc[-1])
                else:
                    for tk in tickers_list:
                        if tk in close_df:
                            current_prices[tk] = float(close_df[tk].iloc[-1])
        except Exception as e:
            print(f"Error downloading batch prices: {e}")
            
    for pf in parsed_files:
        if (pf["is_expired"] and pf["realized_return"] is None) or not pf["is_expired"]:
            current_value = 0.0
            valid = True
            for tk in pf["top_3_tickers"]:
                price = current_prices.get(tk)
                if price is not None and price > 0:
                    current_value += price
                else:
                    valid = False
            
            if valid and pf["initial_cost"] > 0:
                calc_return = (current_value - pf["initial_cost"]) / pf["initial_cost"]
                if pf["is_expired"]:
                    pf["realized_return"] = calc_return
                    try:
                        pf["data_ref"]["realized_return"] = pf["realized_return"]
                        with open(pf["path"], "w") as f:
                            json.dump(pf["data_ref"], f)
                    except:
                        pass
                else:
                    pf["realized_return"] = calc_return

        history_data.append({
            "filename": pf["filename"],
            "timestamp": pf["timestamp"],
            "prob_model": pf["prob_model"],
            "horizon": pf["horizon"],
            "top_3_tickers": pf["top_3_tickers"],
            "is_expired": pf["is_expired"],
            "realized_return": pf["realized_return"]
        })
        
    return {"status": "success", "history": history_data}

@app.get("/api/history/{filename}")
def get_history(filename: str):
    path = os.path.join(history_dir, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                import json
                return json.load(f)
            except:
                pass
    return {"status": "error", "message": "Not found"}

@app.post("/api/history")
async def save_history(request: Request):
    data = await request.json()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{timestamp}.json"
    with open(os.path.join(history_dir, filename), "w") as f:
        import json
        json.dump(data, f)
    return {"status": "success", "filename": filename}

def compute_macro_ai_adjustment(model: str = "gemma4:e4b"):
    try:
        import requests
        import json
        from ddgs import DDGS
        
        macro = get_macro_factors()
        
        # Step 1: Tool Calling / Search Decision
        prompt_1 = f"""You are a quantitative finance AI. Current macroeconomic conditions:
- VIX (Volatility Index): {macro.get('vix')}
- 10-Year Treasury Yield: {macro.get('tnx')}%
- SPY Trend (Z-Score): {macro.get('spy_zscore')}

To score stock sectors accurately, you may need recent financial news. 
If you want to search the web, output a JSON object with a "query" key.
If you do not need to search, output an empty query string.
Example 1: {{"query": "latest federal reserve interest rate news"}}
Example 2: {{"query": ""}}

Output ONLY valid JSON.
"""
        res_search = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt_1,
            "stream": False,
            "format": "json"
        }, timeout=120)
        
        news_context = "No recent news searched."
        if res_search.status_code == 200:
            try:
                search_data = json.loads(res_search.json().get("response", "{}"))
                query = search_data.get("query", "").strip()
                if query:
                    print(f"Agent requested web search: '{query}'")
                    results = DDGS().text(query, max_results=3)
                    news_context = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
                    print(f"Search found {len(results)} results.")
            except Exception as e:
                print(f"Agent search loop failed: {e}")
        
        # Step 2: Final Scoring
        prompt_2 = f"""
Current macroeconomic conditions:
- VIX (Volatility Index): {macro.get('vix')}
- 10-Year Treasury Yield: {macro.get('tnx')}%
- SPY Trend (Z-Score): {macro.get('spy_zscore')}

Recent News Context:
{news_context}

Based on these conditions and news, provide a sentiment multiplier from -1.0 to 1.0 for each of the following stock sectors, where -1.0 means highly negative outlook and 1.0 means highly positive outlook:
Technology, Financial Services, Healthcare, Consumer Cyclical, Industrials, Energy, Utilities, Real Estate, Basic Materials, Communication Services, Consumer Defensive.

Return ONLY a valid JSON dictionary where keys are the sector names and values are the float multipliers.
"""
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt_2,
            "stream": False,
            "format": "json"
        }, timeout=180)
        
        if res.status_code == 200:
            data = res.json()
            response_text = data.get("response", "{}")
            try:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    clean_text = json_match.group(1)
                else:
                    clean_text = response_text.strip()
                
                sector_scores = json.loads(clean_text)
                save_ai_macro_scores(sector_scores, news_context)
                return {"status": "success", "sector_scores": sector_scores, "news_context": news_context}
            except json.JSONDecodeError:
                return {"status": "error", "message": "Failed to parse JSON", "raw": response_text}
        else:
            return {"status": "error", "message": "Ollama API failed", "status_code": res.status_code}
    except Exception as e:
        # Fallback to neutral if Ollama is not running
        print(f"Macro AI Error: {e}")
        return {"status": "error", "message": str(e), "sector_scores": {}}

@app.get("/api/history/ratings/{ticker}")
def get_history_ratings(ticker: str):
    history = get_analyst_rating_history(ticker.upper())
    return {"status": "success", "ticker": ticker, "history": history}

@app.get("/api/history/macro/{sector}")
def get_history_macro(sector: str):
    history = get_ai_macro_score_history(sector)
    return {"status": "success", "sector": sector, "history": history}

@app.get("/api/cointegration")
def get_cointegration(ticker_y: str, ticker_x: str):
    data_y = fetch_data(ticker_y, start="2020-01-01")
    data_x = fetch_data(ticker_x, start="2020-01-01")
    
    if data_y.empty or data_x.empty:
        return {"status": "error", "message": "Failed to fetch data for one or both tickers"}
        
    coint_result = calculate_cointegration(data_y['Close'], data_x['Close'])
    spread = coint_result['spread']
    half_life = estimate_half_life(spread)
    z_scores = generate_z_scores(spread)
    
    return {
        "status": "success",
        "p_value": coint_result['p_value'],
        "hedge_ratio": coint_result['hedge_ratio'],
        "half_life": float(half_life) if not np.isinf(half_life) and not np.isnan(half_life) else None,
        "current_z_score": float(z_scores.iloc[-1]) if not pd.isna(z_scores.iloc[-1]) else 0.0
    }

class FinalPickRequest(BaseModel):
    model: str
    candidates: list

@app.post("/api/final_pick")
def api_final_pick(req: FinalPickRequest):
    import json
    import requests
    
    # Format candidates for the LLM
    candidate_summary = ""
    for c in req.candidates:
        tk = c.get('ticker', 'Unknown')
        sector = c.get('fundamentals', {}).get('sector', 'Unknown')
        price = c.get('current_price', 0)
        rank = c.get('rank_score', 0)
        prob = c.get('biweekly_probs', {}).get('prob_success', 0)
        stop = c.get('biweekly_probs', {}).get('prob_stop', 0)
        val = c.get('value_score', 0)
        z = c.get('z_score', 0)
        rationale = c.get('rationale', 'None provided')
        sentiment_note = c.get('sentiment_note', {})
        sentiment_text = sentiment_note.get('note', 'None') if isinstance(sentiment_note, dict) else 'None'
        
        candidate_summary += f"- **{tk}** ({sector}) | Price: ${price:.2f} | Rank: {rank:.1f}\n"
        candidate_summary += f"  - Upside Prob: {prob*100:.1f}%, Downside Prob: {stop*100:.1f}%\n"
        candidate_summary += f"  - Value Score: {val:.1f}, Trend Z-Score: {z:.2f}\n"
        candidate_summary += f"  - AI Sentiment: {sentiment_text}\n"
        candidate_summary += f"  - Rationale: {rationale}\n\n"

    prompt = f"""You are an expert quantitative portfolio manager. I am giving you a list of top-ranked stock candidates from our internal screener. 
The screener evaluates based on GARCH probabilities, mean-reversion (z-scores), value, AI sentiment analysis, and analyst consensus.

Here are the top candidates:
{candidate_summary}

Based on this data, please select TWO stocks:
1. Your absolute best **Top Pick** (the most solid risk/reward setup).
2. A **Reach Pick** (a higher-risk, potentially higher-reward candidate).

Format your response exactly as follows:

### Top Pick: [TICKER]
[A 1-2 paragraph detailed justification explaining why this stock offers the best solid risk/reward setup based on the metrics and sentiment provided.]

### Reach Pick: [TICKER]
[A 1-2 paragraph detailed justification explaining why this stock is selected as a reach pick based on the metrics and sentiment provided.]

Do not use markdown bolding in the justification bodies.
"""

    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": req.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }, timeout=300)
        if res.status_code == 200:
            return {"status": "success", "pick_markdown": res.json().get("response", "")}
        else:
            return {"status": "error", "message": f"LLM returned {res.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class BacktestRequest(BaseModel):
    ticker: str
    entry_z: float = 2.0
    exit_z: float = 0.0

@app.post("/api/run_backtest")
def run_backtest_endpoint(req: BacktestRequest):
    data = fetch_data(req.ticker, start="2020-01-01")
    if data.empty:
        return {"status": "error", "message": "Data fetch failed"}
        
    # Example mean-reversion signal generation
    # Assume we trade against its own moving average
    prices = data['Close']
    rolling_mean = prices.rolling(window=21).mean()
    rolling_std = prices.rolling(window=21).std()
    z_scores = (prices - rolling_mean) / rolling_std
    
    # Vectorized signal generation
    long_entry = z_scores < -req.entry_z
    long_exit = z_scores > -req.exit_z
    
    short_entry = z_scores > req.entry_z
    short_exit = z_scores < req.exit_z
    
    long_signals = pd.Series(np.nan, index=z_scores.index)
    long_signals[long_entry] = 1
    long_signals[long_exit] = 0
    long_signals = long_signals.ffill().fillna(0)
    
    short_signals = pd.Series(np.nan, index=z_scores.index)
    short_signals[short_entry] = -1
    short_signals[short_exit] = 0
    short_signals = short_signals.ffill().fillna(0)
    
    signals = long_signals + short_signals
        
    res = run_vectorized_backtest(prices, signals)
    if "error" in res:
        return {"status": "error", "message": res["error"]}
        
    return {"status": "success", "result": res}

class PairsBacktestRequest(BaseModel):
    ticker_y: str
    ticker_x: str
    entry_z: float = 2.0
    exit_z: float = 0.0

@app.post("/api/run_pairs_backtest")
def run_pairs_backtest_endpoint(req: PairsBacktestRequest):
    data_y = fetch_data(req.ticker_y, start="2020-01-01")
    data_x = fetch_data(req.ticker_x, start="2020-01-01")
    
    if data_y.empty or data_x.empty:
        return {"status": "error", "message": "Data fetch failed for one or both tickers"}
        
    coint_result = calculate_cointegration(data_y['Close'], data_x['Close'])
    spread = coint_result['spread']
    if spread.empty:
        return {"status": "error", "message": "Spread calculation failed"}
        
    z_scores = generate_z_scores(spread)
    
    # Vectorized signal generation
    long_entry = z_scores < -req.entry_z
    long_exit = z_scores > -req.exit_z
    
    short_entry = z_scores > req.entry_z
    short_exit = z_scores < req.exit_z
    
    long_signals = pd.Series(np.nan, index=z_scores.index)
    long_signals[long_entry] = 1
    long_signals[long_exit] = 0
    long_signals = long_signals.ffill().fillna(0)
    
    short_signals = pd.Series(np.nan, index=z_scores.index)
    short_signals[short_entry] = -1
    short_signals[short_exit] = 0
    short_signals = short_signals.ffill().fillna(0)
    
    signals = long_signals + short_signals
    
    res = run_vectorized_pairs_backtest(data_y['Close'], data_x['Close'], signals)
    
    if "error" in res:
        return {"status": "error", "message": res["error"]}
        
    return {"status": "success", "result": res}

@app.get("/api/universe")
def get_universe_list():
    from backend.database import get_universe
    universe = get_universe()
    return {"status": "success", "universe": universe}

@app.post("/api/universe")
def add_to_universe(req: UniverseAddRequest):
    tk = req.ticker.upper().strip()
    if not tk:
        return {"status": "error", "message": "Ticker cannot be empty"}
        
    try:
        import yfinance as yf
        from backend.database import get_universe, save_universe
        
        # Check if already exists
        universe = get_universe()
        if tk in universe:
            return {"status": "error", "message": f"{tk} is already in the universe."}
            
        # Verify it's a valid ticker
        data = yf.download(tk, period="5d", progress=False)
        if data.empty:
            return {"status": "error", "message": f"Could not fetch data for {tk}. Ticker might be invalid."}
            
        # Add to universe
        universe.append(tk)
        save_universe(universe)
        return {"status": "success", "message": f"Successfully added {tk} to the universe.", "universe": universe}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/api/macro_ai")
def get_macro_ai_adjustment_endpoint(model: str = "gemma4:e4b"):
    try:
        from backend.main import compute_macro_ai_adjustment
        return compute_macro_ai_adjustment(model)
    except Exception as e:
        print(f"Macro AI Queue Error: {e}")
        return {"status": "error", "scores": {}}

@app.get("/api/sortino_baskets")
def get_sortino_baskets():
    try:
        baskets = find_top_sortino_baskets(basket_size=5, num_baskets=5, sample_size=100, iterations=10000)
        return {"status": "success", "baskets": baskets}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/api/performance_chart")
def get_performance_chart(tickers: str, start_date: str):
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not ticker_list:
            return {"status": "error", "message": "No tickers provided"}
            
        # Parse start_date and add 1 day to ensure we capture the start correctly
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", ""))
        except ValueError:
            start_dt = datetime.strptime(start_date.split("T")[0], "%Y-%m-%d")
            
        # Fetch data from fetch_start_dt to today
        fetch_start_dt = start_dt - timedelta(days=60)
        df = yf.download(ticker_list, start=fetch_start_dt.strftime("%Y-%m-%d"), progress=False)
        
        if df.empty or 'Close' not in df:
            return {"status": "error", "message": "No price data found"}
            
        close_df = df['Close']
            
        dates = [d.strftime("%Y-%m-%d") for d in close_df.index]
        
        import pandas as pd
        # Find the base_date (first date in the index >= start_dt)
        base_date = None
        base_index = 0
        target_ts = pd.Timestamp(start_dt.date())
        for i, d in enumerate(close_df.index):
            if d >= target_ts:
                base_date = d
                base_index = i
                break
                
        if base_date is None:
            # Fallback to the last available date if start_dt is in the future
            base_date = close_df.index[-1]
            base_index = len(close_df.index) - 1
            
        base_date_str = base_date.strftime("%Y-%m-%d")
        relative_days = [i - base_index for i in range(len(close_df.index))]
        
        raw_prices = {}
        for tk in ticker_list:
            if tk in close_df:
                # Get series, drop NAs
                series = close_df[tk].dropna()
                if not series.empty:
                    # Align to the main dates index and forward fill
                    aligned = series.reindex(close_df.index).fillna(method='ffill').tolist()
                    raw_prices[tk] = aligned
                    
        return {
            "status": "success",
            "dates": dates,
            "relative_days": relative_days,
            "base_date": base_date_str,
            "base_index": base_index,
            "prices": raw_prices
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import os
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
    # Start the app
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
