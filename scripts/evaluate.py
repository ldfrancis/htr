"""Evaluate TrOCR on an IAM text-line split and report CER and WER.

Transcribes every line in a split of the chosen IAM dataset and appends one row per text
normalization (see htr/metrics.py) to a summary CSV:

    <output_csv>        # model, dataset, split, num_lines, num_beams, normalization, cer, wer
    <predictions_csv>   # optional: image_id, text (prediction), ground_truth

By default (--dataset default) train and validation come from the original IAM lines in
--data-dir and test from the TrOCR authors' release of the IAM test set (Kaggle hub). Pass
--dataset local, trocr or teklia to pick one source for any split, and --num-beams 10 to match
the TrOCR paper's decoding.

Example:
    python scripts/evaluate.py --split test --num-beams 10 --predictions-csv output/predictions_trocr_test_beam10.csv
    python scripts/evaluate.py --split validation --predictions-csv output/predictions_local_validation.csv
    python scripts/evaluate.py --dataset teklia --split test --num-beams 10 --predictions-csv output/predictions_test_beam10.csv
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from htr.data import (  # noqa: E402
    DATA_DIR,
    DATASET_NAME,
    LOCAL_DATASET_NAME,
    TROCR_DATASET_NAME,
    image_id,
    iter_batches,
    load_iam_lines,
    load_local_iam_lines,
    load_split,
    load_trocr_iam_test,
)
from htr.metrics import NORMALIZATIONS, score  # noqa: E402
from htr.model import Recognizer  # noqa: E402
from htr.results import write_csv  # noqa: E402

DATASETS = ["local", "trocr", "teklia"]
FIELDS = ["model", "dataset", "split", "num_lines", "num_beams", "normalization", "cer", "wer"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["default", *DATASETS], default="default",
                        help="default: local for train/validation, trocr for test; teklia: Teklia/IAM-line from Hugging Face; trocr: the TrOCR authors' IAM test set from Kaggle hub (test split only); "
                             "local: the original IAM lines, lines.txt and splits under --data-dir")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="root of the extracted IAM data (lines/, lines.txt, splits/)")
    parser.add_argument("--split", default="test", help="dataset split to evaluate on")
    parser.add_argument("--num-lines", type=int, help="evaluate only the first N lines (default: whole split)")
    parser.add_argument("--batch-size", type=int, default=32, help="lines per forward pass")
    parser.add_argument("--num-beams", type=int, default=1, help="beam size for decoding (1 = greedy)")
    parser.add_argument("--output-csv", type=Path, default=Path("output/evaluation.csv"), help="summary CSV to append to")
    parser.add_argument("--predictions-csv", type=Path, help="optionally save per-line predictions here")
    return parser.parse_args()


def main():
    args = parse_args()
    recognizer = Recognizer()
    if args.dataset == "default":
        dataset, dataset_name = load_split(args.split, num_lines=args.num_lines, data_dir=args.data_dir)
    elif args.dataset == "trocr":
        if args.split != "test":
            raise SystemExit("--dataset trocr only has a test split")
        dataset, dataset_name = load_trocr_iam_test(num_lines=args.num_lines), TROCR_DATASET_NAME
    elif args.dataset == "local":
        dataset, dataset_name = load_local_iam_lines(args.data_dir, args.split, num_lines=args.num_lines), LOCAL_DATASET_NAME
    else:
        dataset, dataset_name = load_iam_lines(args.split, num_lines=args.num_lines), DATASET_NAME

    predictions, references = [], []
    batches = iter_batches(dataset, args.batch_size)
    for _, images, ground_truths in tqdm(batches, total=-(-len(dataset) // args.batch_size), desc=f"Recognizing {args.split}"):
        predictions.extend(recognizer(images, num_beams=args.num_beams))
        references.extend(ground_truths)

    rows = []
    for normalization in NORMALIZATIONS:
        metrics = score(references, predictions, normalization)
        rows.append({
            "model": recognizer.model_name,
            "dataset": dataset_name,
            "split": args.split,
            "num_lines": len(references),
            "num_beams": args.num_beams,
            "normalization": normalization,
            "cer": f"{metrics['cer']:.4f}",
            "wer": f"{metrics['wer']:.4f}",
        })
    write_csv(args.output_csv, rows, FIELDS, append=True)

    if args.predictions_csv:
        prediction_rows = [
            {"image_id": image_id(args.split, i), "text": prediction, "ground_truth": ground_truth}
            for i, (prediction, ground_truth) in enumerate(zip(predictions, references))
        ]
        write_csv(args.predictions_csv, prediction_rows, ["image_id", "text", "ground_truth"])

    print(f"\n{recognizer.model_name} on {dataset_name} [{args.split}, {len(references)} lines, num_beams={args.num_beams}]")
    for row in rows:
        print(f"[{row['normalization']:>12}] CER: {float(row['cer']):.2%}  WER: {float(row['wer']):.2%}")
    print(f"Appended results to {args.output_csv}")


if __name__ == "__main__":
    main()
