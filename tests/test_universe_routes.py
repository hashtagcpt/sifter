import pandas as pd
from starlette.routing import Route

import backend.database as database
import backend.main as main


def test_no_duplicate_post_route_for_api_universe():
    """
    backend/main.py previously registered two @app.post("/api/universe") handlers
    (generate_universe and add_to_universe). FastAPI/Starlette resolves duplicate
    path+method registrations by first match, silently making the second one dead
    code and breaking the "Add Ticker" button in the Universe Manager tab.
    """
    post_universe_routes = [
        r for r in main.app.routes
        if isinstance(r, Route) and r.path == "/api/universe" and "POST" in (r.methods or set())
    ]
    assert len(post_universe_routes) == 1
    assert post_universe_routes[0].endpoint is main.add_to_universe


def test_post_api_universe_adds_ticker(monkeypatch):
    fake_universe = ["AAPL", "MSFT"]
    saved = {}

    monkeypatch.setattr(database, "get_universe", lambda: list(fake_universe))
    monkeypatch.setattr(database, "save_universe", lambda tickers: saved.update(tickers=tickers))
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.DataFrame({"Close": [1.0, 2.0]}))

    result = main.add_to_universe(main.UniverseAddRequest(ticker="zzzz"))

    assert result["status"] == "success"
    assert "ZZZZ" in saved["tickers"]


def test_post_api_universe_rejects_duplicate_ticker(monkeypatch):
    monkeypatch.setattr(database, "get_universe", lambda: ["AAPL", "ZZZZ"])

    result = main.add_to_universe(main.UniverseAddRequest(ticker="zzzz"))

    assert result["status"] == "error"
