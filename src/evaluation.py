import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def report_binary_predictions(y_true, fake_probabilities, title: str) -> None:
    y_true = np.asarray(y_true, dtype=int)
    fake_probabilities = np.asarray(fake_probabilities, dtype=float)
    if y_true.ndim != 1 or fake_probabilities.shape != y_true.shape:
        raise ValueError("Labels and predicted probabilities must be matching 1D arrays.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Expected binary labels: 0 (genuine) and 1 (fake).")
    if not np.isfinite(fake_probabilities).all() or (
        (fake_probabilities < 0) | (fake_probabilities > 1)
    ).any():
        raise ValueError("Predicted fake probabilities must be finite values from 0 to 1.")

    y_pred = (fake_probabilities >= 0.5).astype(int)
    print(title)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["OR (genuine)", "CG (fake)"],
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix [rows=true, columns=predicted; labels 0, 1]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    if np.unique(y_true).size == 2:
        print(f"ROC-AUC: {roc_auc_score(y_true, fake_probabilities):.4f}")
    else:
        print("ROC-AUC unavailable: evaluation labels contain only one class.")
