import pandas as pd
import numpy as np
import pytest
from backend.metrics import calculate_value_score, detrend_and_zscore, calculate_probabilities

def test_calculate_value_score():
    funds = {
        "pe_ratio": 15,
        "price_to_book": 2.0,
        "ev_to_ebitda": 10.0,
    }
    score = calculate_value_score(funds)
    assert isinstance(score, float)

def test_calculate_value_score_empty():
    score = calculate_value_score({})
    assert score == 0.0

def test_detrend_and_zscore():
    data = pd.DataFrame({"Close": np.linspace(100, 200, 150)})
    zscore = detrend_and_zscore(data['Close'])
    assert not zscore.isna().all()

def test_detrend_and_zscore_too_short():
    data = pd.DataFrame({"Close": [1, 2, 3]})
    zscore = detrend_and_zscore(data['Close'])
    assert isinstance(zscore, pd.Series)

def test_calculate_biweekly_probabilities():
    # Generate some realistic looking log returns
    returns = np.random.normal(0.001, 0.02, 300)
    data = pd.DataFrame({"Close": np.exp(np.cumsum(returns)) * 100})

    probs = calculate_probabilities(data['Close'], target_upside=0.05, max_downside=-0.05, horizon_days=14)
    assert "prob_success" in probs
    assert "prob_stop" in probs
    assert 0 <= probs["prob_success"] <= 1
    assert 0 <= probs["prob_stop"] <= 1
