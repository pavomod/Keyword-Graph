from __future__ import annotations


def keyword_f1(predicted: str, ground_truth: str) -> float:
    pred_set = {k.strip().lower() for k in predicted.split(",") if k.strip()}
    truth_set = {k.strip().lower() for k in ground_truth.split(",") if k.strip()}

    if not pred_set and not truth_set:
        return 1.0

    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def mean_keyword_f1(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    if not predictions:
        return 0.0

    scores = [keyword_f1(pred, ref) for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores)
