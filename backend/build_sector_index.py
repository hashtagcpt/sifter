import json
import os
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from backend.data_engine import build_universe

SECTOR_FILE = os.path.join(os.path.dirname(__file__), "sector_index.json")

def search_sector_fallback(ticker):
    print(f"  Performing web search fallback for {ticker} sector...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        # Try StockAnalysis.com
        res = requests.get(f"https://stockanalysis.com/stocks/{ticker.lower()}/", headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            # The sector is usually in a table or profile section
            # For simplicity, search text for "Sector:"
            for el in soup.find_all(['td', 'th', 'div', 'span', 'p']):
                if 'Sector' in el.text and len(el.text) < 50:
                    sibling = el.find_next_sibling()
                    if sibling: return sibling.text.strip()
                    parent = el.parent
                    if parent and len(parent.text) < 100:
                        parts = parent.text.split('Sector')
                        if len(parts) > 1:
                            return parts[1].replace(':', '').strip().split('\n')[0]
                            
        # Try Yahoo Finance Profile
        res = requests.get(f"https://finance.yahoo.com/quote/{ticker}/profile", headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            for p in soup.find_all('p'):
                text = p.text
                if 'Sector' in text and 'Industry' in text:
                    parts = text.split('Sector(s)')
                    if len(parts) > 1:
                        sec = parts[1].split('Industry')[0].replace(':', '').strip()
                        if sec: return sec
    except Exception as e:
        print("  Fallback error:", e)
    
    return "Unknown Sector"

def build_index():
    print("Building Universe Sector Index...")
    if os.path.exists(SECTOR_FILE):
        with open(SECTOR_FILE, 'r') as f:
            index = json.load(f)
    else:
        index = {}

    uni = build_universe(max_size=250)
    for tk in uni:
        # Check if already in index and valid
        if tk in index and index[tk] not in ["Unknown Sector", None, "", "Unknown"]:
            continue
            
        print(f"Checking {tk}...")
        sector = None
        try:
            ticker = yf.Ticker(tk)
            sector = ticker.info.get("sector")
            if not sector:
                sector = search_sector_fallback(tk)
        except Exception:
            sector = search_sector_fallback(tk)
            
        index[tk] = sector
        print(f"  -> {sector}")
        
        # Save incrementally
        with open(SECTOR_FILE, 'w') as f:
            json.dump(index, f, indent=2)
            
    print("Sector index build complete.")

if __name__ == "__main__":
    build_index()
