import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score


def macro_auc(y_true, y_prob):
    # One-vs-rest ROC AUC, averaged over the classes that actually occur in y_true.
    y_true = np.asarray(y_true)
    aucs = []
    for c in np.unique(y_true):
        positives = y_true == c
        if positives.all():
            continue
        aucs.append(roc_auc_score(positives, y_prob[:, c]))
    return float(np.mean(aucs)) if aucs else None


def classification_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if y_prob is not None:
        metrics["auc"] = macro_auc(y_true, y_prob)
    metrics["confusion_matrix"] = cm
    return metrics


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
