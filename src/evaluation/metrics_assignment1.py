
import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)


def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + fn + 1e-8)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    return np.array([[tn, fp],
                     [fn, tp]])
    
def classification_report(y_true, y_pred):
    acc = accuracy(y_true, y_pred)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    report = f"""
Accuracy : {acc:.4f}
Precision: {p:.4f}
Recall   : {r:.4f}
F1-score : {f1:.4f}
"""
    return report
