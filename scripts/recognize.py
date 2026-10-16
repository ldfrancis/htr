"""Recognize a batch of IAM text lines with TrOCR and save the results to disk.

Transcribes lines from a split (original IAM lines for train/validation, the TrOCR release for
test; see htr.data.load_split) and writes:

    <output_dir>/
        results.csv   # image_id, text (prediction), ground_truth
        img/
            <image_id>.png

Example:
    python scripts/recognize.py --split train --num-lines 32 --batch-size 8 --output-dir output
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from htr.data import DATA_DIR, image_id, iter_batches, load_split  # noqa: E402
from htr.model import Recognizer  # noqa: E402
from htr.results import write_csv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", help="dataset split to read lines from")
    parser.add_argument("--start", type=int, default=0, help="index of the first line to use")
    parser.add_argument("--num-lines", type=int, default=32, help="number of lines to recognize")
    parser.add_argument("--batch-size", type=int, default=8, help="lines per forward pass")
    parser.add_argument("--num-beams", type=int, default=1, help="beam size for decoding (1 = greedy)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="root of the extracted IAM data (lines/, lines.txt, splits/)")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="where to write results")
    return parser.parse_args()


def main():
    args = parse_args()
    recognizer = Recognizer()
    dataset, _ = load_split(args.split, start=args.start, num_lines=args.num_lines, data_dir=args.data_dir)

    img_dir = args.output_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for offset, images, ground_truths in iter_batches(dataset, args.batch_size):
        predictions = recognizer(images, num_beams=args.num_beams)
        for i, (image, prediction, ground_truth) in enumerate(zip(images, predictions, ground_truths)):
            line_id = image_id(args.split, args.start + offset + i)
            image.save(img_dir / f"{line_id}.png")
            rows.append({"image_id": line_id, "text": prediction, "ground_truth": ground_truth})
            print(f"{line_id}: {prediction}")

    csv_path = args.output_dir / "results.csv"
    write_csv(csv_path, rows, fieldnames=["image_id", "text", "ground_truth"])
    print(f"\nSaved {len(rows)} results to {csv_path} and images to {img_dir}/")


if __name__ == "__main__":
    main()
