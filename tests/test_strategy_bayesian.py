import pytest
from backend.strategy_bayesian import apply_bayesian_reweighting

def test_apply_bayesian_reweighting():
    base_weights = {
        "AAPL": 0.6,
        "MSFT": 0.4
    }
    views = {"AAPL": 0.7, "MSFT": 0.4}
    confidences = {"AAPL": 0.8, "MSFT": 0.5}

    weights = apply_bayesian_reweighting(base_weights, views, confidences)
    
    assert "AAPL" in weights
    assert "MSFT" in weights
    
    # Check if sum is close to 1
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-5

def test_apply_bayesian_reweighting_no_results():
    weights = apply_bayesian_reweighting({}, {}, {})
    assert weights == {}
