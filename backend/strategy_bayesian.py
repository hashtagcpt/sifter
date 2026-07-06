import numpy as np

def update_probability(prior_prob: float, subjective_prob: float, confidence: float = 0.5) -> float:
    """
    Simple Bayesian updating of a quantitative prior using a subjective view.
    Confidence (0 to 1) weights the subjective probability.
    """
    prior_odds = prior_prob / (1 - prior_prob + 1e-6)
    subj_odds = subjective_prob / (1 - subjective_prob + 1e-6)
    
    blended_odds = prior_odds**(1-confidence) * subj_odds**confidence
    posterior_prob = blended_odds / (1 + blended_odds)
    return posterior_prob

def apply_bayesian_reweighting(base_weights: dict, user_views: dict, confidences: dict) -> dict:
    """
    Given a dictated baseline weight/allocation, adjust it according to subjective views.
    user_views: dict of ticker -> subjective positive expected return probability (0.0 to 1.0)
    confidences: dict of ticker -> user's confidence in their view (0.0 to 1.0)
    """
    new_weights = {}
    total_base = sum(base_weights.values()) if base_weights else 1.0
    
    adjusted_scores = {}
    for ticker, weight in base_weights.items():
        base_prob = weight / total_base
        
        if ticker in user_views:
            subj_prob = user_views[ticker]
            conf = confidences.get(ticker, 0.5)
            adjusted_scores[ticker] = update_probability(base_prob, subj_prob, conf)
        else:
            adjusted_scores[ticker] = base_prob
            
    total_score = sum(adjusted_scores.values())
    
    if total_score > 0:
        for ticker, score in adjusted_scores.items():
            new_weights[ticker] = score / total_score
    else:
        new_weights = base_weights.copy()
        
    return new_weights
