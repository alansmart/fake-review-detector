""""Explainable Trust Score built on top of DML model + interpretable signals."""

def _level(score: float) -> str:
    return "High" if score >= 70 else "Medium" if score >= 40 else "Low"

def _interpretable_part(feats: dict) -> dict:
    consistency = 1 - min(feats["rating_sentiment_gap"] / 2, 1)
    style = 1 - min(feats["exclaim_ratio"] * 20 + feats["caps_ratio"] * 5, 1)
    return {"consistency": consistency, "style": style}

def _reasons(p_fake: float, parts: dict) -> list:
    reasons = []
    if p_fake > 0.5:
        reasons.append("Language pattern resembles generated or fake reviews")
    if parts["consistency"] < 0.5:
        reasons.append("Rating does not match review sentiment")
    if parts["style"] < 0.5:
        reasons.append("Excessive exclamation/capital letters")
    return reasons

def trust_score(p_fake: float, feats: dict, reviewer_history: dict | None = None) -> dict:
    """Trust score for baseline/fusion models (0-100)."""
    parts = {"model": 1 - p_fake, **_interpretable_part(feats)}
    weights = {"model": 0.5, "consistency": 0.3, "style": 0.2}

    score = 100 * sum(weights[k] * parts[k] for k in weights)
    return {"trust_score": round(score, 1), "level": _level(score), "p_fake": round(float(p_fake), 4), "components": {k: round(float(v), 3) for k, v in parts.items()},
            "reasons": _reasons(p_fake, parts)}

def dml_trust_score(p_fake: float, margin: float, novelty: float, feats: dict, novelty_threshold: float = 0.35) -> dict:
    """Trust score for the DML model, with out-of-distribution abtention."""
    if novelty_threshold <= 0:
        raise ValueError("novelty_threshold must be greater than zero.")

    parts = {
        "model": 1 - p_fake,
        "certainty": min(margin / 0.5, 1),
        "familiarity": 1 - min(novelty / novelty_threshold, 1),
        **_interpretable_part(feats),
    }
    weights = {
        "model": 0.5,
        "certainty": 0.15,
        "familiarity": 0.1,
        "consistency": 0.15,
        "style": 0.1,
    }
    score = 100 * sum(weights[k] * parts[k] for k in weights)

    reasons = _reasons(p_fake, parts)
    level = _level(score)
    if novelty > novelty_threshold:
        level = "Uncertain"
        reasons.append("Review differs from training examples; manual review is advised")

    return {
        "trust_score": round(score, 1),
        "level": level,
        "p_fake": round(float(p_fake), 4),
        "components": {k: round(float(v), 3) for k, v in parts.items()},
        "reasons": reasons,
    }


__all__ = ["trust_score", "dml_trust_score"]