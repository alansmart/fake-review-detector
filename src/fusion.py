"""Multimodal fusion: DistilBERT embeddings + metadata features -> MLP."""

from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"
device = "cuda" if torch.cuda.is_available() else "cpu"

from evaluation import report_binary_predictions


class FusionNet(nn.Module):
    """MLP for multimodal fusion of text embeddings and metadata features."""

    def __init__(self, text_dim=768, meta_dim=11):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(text_dim, 128), nn.ReLU(), nn.Dropout(0.3))
        self.meta = nn.Sequential(nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.3))
        self.comb = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, text, meta):
        return self.comb(torch.cat([self.text(text), self.meta(meta)], dim=1)).squeeze(1)


def to_t(a):
    return torch.tensor(a, device=device, dtype=torch.float32)


def train_model(T, M, y, train_idx, epochs, batch_size, lr):
    scaler = StandardScaler().fit(M[train_idx])
    scaled_M = scaler.transform(M)
    Tt, Mt, yt = to_t(T), to_t(scaled_M), to_t(y)

    model = FusionNet(T.shape[1], M.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        perm = np.random.permutation(train_idx)
        for i in range(0, len(perm), batch_size):
            b = torch.tensor(perm[i:i + batch_size], device=device)
            loss = lossf(model(Tt[b], Mt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(b)
            total += len(b)
    return model, scaler


def predict_probabilities(model, scaler, T, M, indices):
    model.eval()
    text = to_t(T[indices])
    metadata = to_t(scaler.transform(M[indices]))
    with torch.no_grad():
        return torch.sigmoid(model(text, metadata)).cpu().numpy()


def main(epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    torch.manual_seed(42)
    np.random.seed(42)
    MODELS.mkdir(parents=True, exist_ok=True)

    T = np.load(PROC / "T.npy")
    M = np.load(PROC / "M.npy")
    y = np.load(PROC / "y.npy")
    if not (len(T) == len(M) == len(y)):
        raise ValueError("Text embeddings, metadata features, and labels must have equal row counts.")

    class_counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("At least two examples per class are needed for stratified evaluation.")

    probabilities = np.empty(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, validation_idx) in enumerate(cv.split(T, y), start=1):
        model, scaler = train_model(
            T, M, y, train_idx, epochs, batch_size, lr
        )
        probabilities[validation_idx] = predict_probabilities(
            model, scaler, T, M, validation_idx
        )
        print(f"Completed fold {fold}/{n_splits}")

    report_binary_predictions(
        y, probabilities, f"{n_splits}-fold stratified out-of-fold evaluation"
    )

    all_indices = np.arange(len(y))
    final_model, final_scaler = train_model(
        T, M, y, all_indices, epochs, batch_size, lr
    )
    torch.save(final_model.state_dict(), MODELS / "fusion.pt")
    joblib.dump(final_scaler, MODELS / "scaler.joblib")
    print("Final model refit on all supplied data and saved to", MODELS / "fusion.pt")

if __name__ == "__main__":
    main()
        