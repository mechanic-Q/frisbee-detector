"""Compare classifier results against human labels, output confusion matrix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-csv", required=True, help="Human labels CSV (review_results.csv)")
    parser.add_argument("--classifier-csv", required=True, help="Classifier output CSV")
    parser.add_argument("--human-label-col", default="result", help="Column name for human label")
    parser.add_argument("--human-positive", default="TP", help="Positive label value in human CSV")
    parser.add_argument("--classifier-label-col", default="label", help="Column name for classifier label")
    parser.add_argument("--classifier-positive", default="frisbee", help="Positive label value in classifier CSV")
    args = parser.parse_args()

    human = {}
    with open(args.human_csv) as f:
        for row in csv.DictReader(f):
            human[row["filename"]] = row[args.human_label_col].strip().upper() == args.human_positive.upper()

    classifier = {}
    with open(args.classifier_csv) as f:
        for row in csv.DictReader(f):
            classifier[row["filename"]] = row[args.classifier_label_col] == args.classifier_positive

    tp = fp = fn = tn = 0
    for fname, is_positive in human.items():
        if fname not in classifier:
            continue
        pred = classifier[fname]
        if is_positive and pred:
            tp += 1
        elif not is_positive and pred:
            fp += 1
        elif is_positive and not pred:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1)

    print(f"=== Classification Report ===")
    print(f"Total:       {total}")
    print(f"TP (correct frisbee):     {tp}")
    print(f"FP (false frisbee):       {fp}")
    print(f"FN (missed frisbee):      {fn}")
    print(f"TN (correct non-frisbee): {tn}")
    print(f"Accuracy:    {accuracy:.1f}%")
    print(f"Precision:   {precision:.1f}%")
    print(f"Recall:      {recall:.1f}%")
    print(f"F1:          {f1:.1f}%")


if __name__ == "__main__":
    main()
