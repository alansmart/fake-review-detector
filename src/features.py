import re
import numpy as np
import pandas as pd
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

FEATURE_COLS = [
    "length", "word_count", "avg_word_len", "exclaim_ratio", "caps_ratio",
    "first_person_ratio", "sentiment", "rating", "rating_sentiment_gap",
    "readability", "unique_word_ratio"
]

def extract_features(text: str, rating: float) -> dict:
    text = str(text)
    words = re.findall(r"\w+", text.lower())
    n = max(len(words), 1)
    sentiment = _sia.polarity_scores(text)["compound"]
    norm_rating = (float(rating) - 3) / 2
    return {
        "length": len(text),
        "word_count": len(words),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "exclaim_ratio": text.count("!") / max(len(text), 1),
        "caps_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
        "first_person_ratio": sum(w in {"i", "me", "my", "mine"} for w in words) / n,
        "sentiment": sentiment,
        "rating": float(rating),
        "rating_sentiment_gap": abs(norm_rating - sentiment),
        "readability": textstat.flesch_reading_ease(text) if words else 0.0,
        "unique_word_ratio": len(set(words)) / n,
    }

def build_feature_frame(df: pd.DataFrame,
                        text_col: str = "text_",
                        rating_col: str = "rating") -> pd.DataFrame:
    """Build the metadata/linguistic feature matrix for a whole DataFrame."""
    rows = [extract_features(t, r) for t, r, in zip(df[text_col], df[rating_col])]
    return pd.DataFrame(rows)[FEATURE_COLS]

__all__ = ["FEATURE_COLS", "extract_features", "build_feature_frame"]