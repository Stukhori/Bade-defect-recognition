import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from windblade.evaluation.metrics import classification_metrics


def test_metrics_match_sklearn_and_preserve_class_order():
    names = ["a", "b", "c"]
    true = np.asarray([0, 0, 1, 1, 2, 2])
    predicted = np.asarray([0, 1, 1, 1, 0, 2])
    result = classification_metrics(true, predicted, names)
    assert result["accuracy"] == 4 / 6
    assert result["balanced_accuracy"] == balanced_accuracy_score(true, predicted)
    assert result["macro_precision"] == precision_score(true, predicted, average="macro", zero_division=0)
    assert result["macro_recall"] == recall_score(true, predicted, average="macro", zero_division=0)
    assert result["macro_f1"] == f1_score(true, predicted, average="macro", zero_division=0)
    assert result["class_order"] == names
    assert result["confusion_matrix_counts"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_missing_predictions_use_explicit_zero_division():
    result = classification_metrics([0, 1, 2], [0, 0, 0], ["a", "b", "c"])
    assert result["zero_division"] == 0
    assert result["per_class"]["b"]["precision"] == 0
    assert result["per_class"]["c"]["f1"] == 0
