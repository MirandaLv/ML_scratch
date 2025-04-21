
import numpy as np

def classification_report(y_true, y_pred, labels=None):
    # for cases that in y_true missing certain labels
    if labels==None:
        labels = sorted(set(y_true + y_pred))

    report = {}
    for label in labels:
        tp = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        case_sum_support = sum(yt == label for yt in y_true)

        report[label] = {"precision": precision,
                         "recall": recall,
                         "f1": f1,
                         "support": case_sum_support}

    overall_recall = sum(v["recall"] for v in report.values()) / len(labels)
    overall_precision = sum(v["precision"] for v in report.values()) / len(labels)
    overall_f1 = sum(v["f1"] for v in report.values()) / len(labels)
    overall_support = sum(v["support"] for v in report.values())

    report["avg / total"] = {
        "precision": round(overall_precision, 2),
        "recall": round(overall_recall, 2),
        "f1-score": round(overall_f1, 2),
        "support": overall_support
    }

    return report


# Example usage
y_true = [1, 2, 2, 4, 5, 1, 2, 3, 4, 5]
y_pred = [1, 2, 2, 4, 5, 1, 3, 3, 4, 5]

report = classification_report(y_true, y_pred)
for label, metrics in report.items():
    print(f"{label}: {metrics}")