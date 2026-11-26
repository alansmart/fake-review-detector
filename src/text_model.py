import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "distilbert-base-uncased"

device = "cuda" if torch.cuda.is_available() else "cpu"

_tok = None
_enc = None

def _load():
    """Load the tokenizer and mpdel once, on first use (lazy loading)."""
    global _tok, _enc
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _enc = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return _tok, _enc

@torch.no_grad()
def embed_texts(texts, batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """Return mean.pooled DistilBERT embeddings with shape [N, 768]."""

    tok, enc = _load()
    texts = [str(t) for t in texts]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length = max_length, return_tensors="pt").to(device)
        h = enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
        if(i // batch_size) % 50 == 0:
            print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(out)

__all__ = ["embed_texts", "MODEL_NAME"]
