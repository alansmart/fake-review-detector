"""Baseline TF-IDF + Logistic Regression (text only)."""

from pathlib  import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
RAW =ROOT / "data" / "raw" / "fake_review_Dataset.csv"
MODELS = ROOT / "models"

from data_utils import load_review_dataset
from evaluation import report_binary_predictions


def load_data():
    return load_review_dataset(RAW)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} reviews | fake ratio = {df['y'].mean():.2%}")
    class_counts = df["y"].value_counts()
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("At least two examples per class are needed for stratified evaluation.")

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        pipe, df["text_"], df["y"], cv=cv, method="predict_proba"
    )[:, 1]
    report_binary_predictions(
        df["y"], probabilities, f"{n_splits}-fold stratified out-of-fold evaluation"
    )

    pipe.fit(df["text_"], df["y"])
    joblib.dump(pipe, MODELS / "baseline.joblib")
    print("Refit on all supplied data. Saved model to", MODELS / "baseline.joblib")


if __name__ == "__main__":
    main()
