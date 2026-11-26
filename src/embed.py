import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from data_utils import load_review_dataset
from features import build_feature_frame
from text_model import embed_texts

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW = PROJECT_DIR / "data" / "raw" / "fake_review_Dataset.csv"
OUT = PROJECT_DIR / "data" / "processed"

if __name__ == "__main__":
    if not RAW.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {RAW}. "
            "Place fake_review_Dataset.csv in the project's data/raw folder."
        )

    os.makedirs(OUT, exist_ok=True)
    df = load_review_dataset(RAW)
    y = df["y"].values
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    np.save(f"{OUT}/T.npy", embed_texts(df["text_"].tolist()))
    np.save(f"{OUT}/M.npy", build_feature_frame(df).values)
    np.save(f"{OUT}/y.npy", y)
    np.save(f"{OUT}/idx_tr.npy", idx_tr)
    np.save(f"{OUT}/idx_te.npy", idx_te)
    print("Saved embeddings:", len(y), "reviews")

    