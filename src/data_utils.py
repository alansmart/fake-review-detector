from pathlib import Path

import pandas as pd


def load_review_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Review dataset not found: {path}")

    df = pd.read_csv(path)
    required = {"text_", "label", "rating"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = df.dropna(subset=["text_"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    invalid_labels = sorted(set(df["label"]) - {"CG", "OR"})
    if invalid_labels:
        raise ValueError(
            f"Unsupported labels: {invalid_labels}. Expected only 'CG' and 'OR'."
        )

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    invalid_ratings = df["rating"].isna() | ~df["rating"].between(1, 5)
    if invalid_ratings.any():
        raise ValueError("Ratings must be numeric values from 1 to 5.")
    if df.empty:
        raise ValueError("Dataset contains no reviews with non-empty text.")

    df["y"] = (df["label"] == "CG").astype(int)
    return df.reset_index(drop=True)
