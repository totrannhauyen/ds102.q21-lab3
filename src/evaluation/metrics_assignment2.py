

from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    classification_report as sk_classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)


def f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1)


def confusion_matrix(y_true, y_pred):
    return sk_confusion_matrix(y_true, y_pred)


def classification_report(y_true, y_pred):
    return sk_classification_report(y_true, y_pred)


def evaluate(y_true, y_pred):

    print("------------ Assignment 2 ------------")
    print("📊 MODEL EVALUATION")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
