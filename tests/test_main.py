import pytest
from fastapi.testclient import TestClient
import pandas as pd
from unittest.mock import patch, MagicMock
import backend.metrics  # Load arch/matplotlib before mocking subprocess.Popen.

# Since main.py starts ollama on import and we don't want to actually start it in tests,
# mock only subprocess.Popen so dependencies still see the real subprocess module.
with patch("subprocess.Popen", MagicMock()):
    from backend.main import app, build_ticker_sentiment_note

client = TestClient(app)

def test_read_root():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Quant Engine Running"}

@patch("backend.main.build_universe")
def test_generate_universe(mock_build):
    # POST /api/universe is the ticker-add endpoint (add_to_universe); building/refreshing
    # the full universe list lives at its own path, POST /api/universe/build, so the two
    # POST handlers that used to collide on "/api/universe" don't shadow each other.
    mock_build.return_value = ["AAPL", "MSFT"]
    response = client.post("/api/universe/build", json={"max_size": 2500})
    assert response.status_code == 200
    assert response.json()["universe"] == ["AAPL", "MSFT"]

@patch("backend.database.get_universe")
@patch("backend.database.save_universe")
@patch("yfinance.download")
def test_add_to_universe(mock_download, mock_save, mock_get_universe):
    mock_get_universe.return_value = ["MSFT"]
    mock_download.return_value = pd.DataFrame({"Close": [100.0]})
    response = client.post("/api/universe", json={"ticker": "aapl"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "AAPL" in data["universe"]

@patch("backend.main.build_universe")
def test_scrape_universe(mock_build):
    mock_build.return_value = ["AAPL", "GOOGL"]
    response = client.post("/api/scrape_universe", json={"max_size": 10})
    assert response.status_code == 200
    assert response.json()["count"] == 2

@patch("backend.main.get_fundamental_metrics")
@patch("backend.main.calculate_value_score")
@patch("backend.main.fetch_analyst_ratings")
@patch("backend.main.fetch_data")
@patch("backend.main.detrend_and_zscore")
@patch("backend.metrics.calculate_probabilities")
@patch("backend.main.generate_rationale")
def test_analyze_ticker(
    mock_rationale, mock_garch, mock_zscore, mock_fetch, 
    mock_ratings, mock_score, mock_funds
):
    mock_funds.return_value = {"marketCap": 1e9}
    mock_score.return_value = 5.0
    mock_ratings.return_value = {"buy": 10, "sell": 1}
    mock_fetch.return_value = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    mock_zscore.return_value = 1.2
    mock_garch.return_value = {"prob_success": 0.6, "prob_stop": 0.2, "exp_return": 0.03}
    mock_rationale.return_value = "Looks good"
    response = client.get("/api/analyze_ticker/AAPL?horizon=14&upside=0.05&downside=-0.05&news_notes=false")
    assert response.status_code == 200
    
    data = response.json()
    assert data.get("status") == "success", data
    res_data = data["data"]
    assert res_data["ticker"] == "AAPL"
    assert "rank_score" in res_data
    assert res_data["biweekly_probs"]["prob_success"] == 0.6
    assert res_data["sentiment_note"] is None


@patch("yfinance.Ticker")
@patch("backend.main._save_cached_sector_note")
@patch("backend.main._load_cached_sector_note")
@patch("requests.post")
@patch("ddgs.DDGS")
def test_build_ticker_sentiment_note(mock_ddgs, mock_post, mock_cache_load, mock_cache_save, mock_yf):
    mock_yf.return_value.news = []
    mock_cache_load.return_value = None
    mock_ddgs.return_value.text.return_value = [
        {
            "title": "Example product demand",
            "href": "https://example.com/aapl-demand",
            "body": "AAPL demand is improving, but services regulation remains a risk.",
        }
    ]
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "response": """
        {
          "sentiment": "Bullish",
          "rationale": "AAPL has a credible demand catalyst, but services regulation could weigh on multiples."
        }
        """
    }

    note = build_ticker_sentiment_note("AAPL", model="test-model")

    assert note["status"] == "success"
    assert "credible demand catalyst" in note["sentence"]
    assert note["sources"][0]["url"] == "https://example.com/aapl-demand"
