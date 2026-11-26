import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluation import report_binary_predictions

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DMLNet(nn.Module):
    def __init__(self, text_dim=768, meta_dim=11, emb_dim=128):
        super().__init__()
        self.text = nn.Sequential(
            nn.Linear(text_dim, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.ReLU(), nn.Dropout(0.3)
        )
        self.comb = nn.Sequential(
            nn.Linear(320, 256), nn.ReLU(), nn.Linear(256, emb_dim)
        )

    def forward(self, text, meta):
        combined = torch.cat([self.text(text), self.meta(meta)], dim=1)
        return F.normalize(self.comb(combined), dim=1)


def supcon_loss(embeddings, labels, temperature=0.1):
    """Supervised contrastive loss over normalized embeddings."""
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected a 2D embedding tensor and 1D labels.")
    if len(embeddings) != len(labels):
        raise ValueError("Embedding and label batch sizes must match.")
    if len(labels) < 2:
        return embeddings.sum() * 0.0
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positive_counts = positives.sum(dim=1)
    valid = positive_counts > 0
    if not valid.any():
        return embeddings.sum() * 0.0

    per_example = -(log_probabilities.masked_fill(~positives, 0).sum(dim=1))
    per_example = per_example / positive_counts.clamp_min(1)
    return per_example[valid].mean()


class DMLScorer:
    """Class prototypes plus nearest-neighbor similarity for DML inference."""

    def __init__(self, Z_train, y_train, tau=0.1, k=10):
        self.Z = F.normalize(torch.as_tensor(Z_train, dtype=torch.float32), dim=1)
        self.y = torch.as_tensor(y_train, dtype=torch.long, device=self.Z.device).view(-1)
        self.tau = tau
        self.k = k
        if self.Z.ndim != 2 or len(self.Z) != len(self.y) or len(self.Z) == 0:
            raise ValueError("Training embeddings and labels must be non-empty and aligned.")
        if tau <= 0 or k <= 0:
            raise ValueError("tau and k must be greater than zero.")

        self.classes = torch.unique(self.y).sort().values
        if not torch.equal(self.classes, torch.tensor([0, 1], device=self.y.device)):
            raise ValueError("DML scoring requires training examples for labels 0 and 1.")
        prototypes = [
            self.Z[self.y == label].mean(dim=0) for label in self.classes
        ]
        self.prototypes = F.normalize(torch.stack(prototypes), dim=1)
        self.fake_index = int(torch.nonzero(self.classes == 1, as_tuple=True)[0].item())

    @torch.no_grad()
    def score(self, embeddings):
        embeddings = F.normalize(
            torch.as_tensor(embeddings, dtype=torch.float32, device=self.Z.device),
            dim=1,
        )
        similarities = embeddings @ self.prototypes.T
        probabilities = torch.softmax(similarities / self.tau, dim=1)
        p_fake = probabilities[:, self.fake_index]
        genuine_index = 1 - self.fake_index
        margin = (
            similarities[:, self.fake_index] - similarities[:, genuine_index]
        ).abs()

        neighbor_count = min(self.k, len(self.Z))
        neighbor_similarity = (embeddings @ self.Z.T).topk(
            neighbor_count, dim=1
        ).values.mean(dim=1)
        novelty = 1 - neighbor_similarity
        return p_fake, margin, novelty


def to_tensor(values):
    return torch.tensor(values, device=DEVICE, dtype=torch.float32)


def train_model(text, metadata, labels, train_indices, epochs, batch_size, learning_rate):
    scaler = StandardScaler().fit(metadata[train_indices])
    text_tensor = to_tensor(text)
    metadata_tensor = to_tensor(scaler.transform(metadata))
    label_tensor = torch.tensor(labels, device=DEVICE, dtype=torch.long)

    model = DMLNet(text.shape[1], metadata.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    for epoch in range(epochs):
        model.train()
        permutation = np.random.permutation(train_indices)
        total_loss = 0.0
        total_samples = 0
        for start in range(0, len(permutation), batch_size):
            batch_indices = permutation[start:start + batch_size]
            batch = torch.tensor(batch_indices, device=DEVICE)
            loss = supcon_loss(
                model(text_tensor[batch], metadata_tensor[batch]),
                label_tensor[batch],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_indices)
            total_samples += len(batch_indices)

    return model, scaler


def predict_fold(model, scaler, text, metadata, labels, train_indices, validation_indices):
    model.eval()
    text_tensor = to_tensor(text)
    metadata_tensor = to_tensor(scaler.transform(metadata))
    label_tensor = torch.tensor(labels, device=DEVICE, dtype=torch.long)
    train_tensor = torch.tensor(train_indices, device=DEVICE)
    validation_tensor = torch.tensor(validation_indices, device=DEVICE)
    with torch.no_grad():
        train_embeddings = model(
            text_tensor[train_tensor], metadata_tensor[train_tensor]
        )
        validation_embeddings = model(
            text_tensor[validation_tensor], metadata_tensor[validation_tensor]
        )
        scorer = DMLScorer(train_embeddings, label_tensor[train_tensor])
        p_fake, _, novelty = scorer.score(validation_embeddings)
    return p_fake.cpu().numpy(), novelty.cpu().numpy()


def main(epochs: int = 20, batch_size: int = 64, learning_rate: float = 1e-3):
    torch.manual_seed(42)
    np.random.seed(42)
    MODELS.mkdir(parents=True, exist_ok=True)

    required = ("T.npy", "M.npy", "y.npy")
    missing = [name for name in required if not (PROC / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data in {PROC}: {', '.join(missing)}. "
            "Run src/embed.py first."
        )

    text = np.load(PROC / "T.npy")
    metadata = np.load(PROC / "M.npy")
    labels = np.load(PROC / "y.npy").astype(int)
    if not (len(text) == len(metadata) == len(labels)):
        raise ValueError("Text, metadata, and label arrays must have equal row counts.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary labels: 0 (genuine) and 1 (fake).")

    class_counts = np.bincount(labels, minlength=2)
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("At least two examples per class are needed for stratified evaluation.")

    oof_probabilities = np.empty(len(labels), dtype=float)
    oof_novelty = np.empty(len(labels), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_indices, validation_indices) in enumerate(
        cv.split(text, labels), start=1
    ):
        model, scaler = train_model(
            text, metadata, labels, train_indices, epochs, batch_size, learning_rate
        )
        probabilities, novelty = predict_fold(
            model, scaler, text, metadata, labels, train_indices, validation_indices
        )
        oof_probabilities[validation_indices] = probabilities
        oof_novelty[validation_indices] = novelty
        print(f"Completed DML fold {fold}/{n_splits}")

    report_binary_predictions(
        labels,
        oof_probabilities,
        f"{n_splits}-fold stratified DML out-of-fold evaluation",
    )
    novelty_threshold = float(np.percentile(oof_novelty, 95))
    print(f"Out-of-fold novelty 95th percentile: {novelty_threshold:.4f}")

    all_indices = np.arange(len(labels))
    model, scaler = train_model(
        text, metadata, labels, all_indices, epochs, batch_size, learning_rate
    )
    model.eval()
    text_tensor = to_tensor(text)
    metadata_tensor = to_tensor(scaler.transform(metadata))
    label_tensor = torch.tensor(labels, device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        train_embeddings = model(text_tensor, metadata_tensor)

    torch.save(model.state_dict(), MODELS / "dml_model.pt")
    torch.save(
        {
            "Z": train_embeddings.cpu(),
            "y": label_tensor.cpu(),
        },
        MODELS / "dml_index.pt",
    )
    joblib.dump(scaler, MODELS / "dml_scaler.joblib")
    with (MODELS / "dml_novelty_threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": novelty_threshold}, f, indent=2)
    print("Saved DML model and inference artifacts to", MODELS)


if __name__ == "__main__":
    main()
