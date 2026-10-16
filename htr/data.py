"""Loading and batching IAM text lines."""

from pathlib import Path

from datasets import Dataset, Image, load_dataset

DATASET_NAME = "Teklia/IAM-line"
TROCR_DATASET_NAME = "changheonkim/iam-trocr"
LOCAL_DATASET_NAME = "IAM (local)"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_split(split, start=0, num_lines=None, data_dir=DATA_DIR):
    """Load a split of the project's data: the original IAM lines (from data_dir) for train and
    validation, and the TrOCR authors' release of the IAM test set (from Kaggle hub) for test.

    Returns (dataset, dataset_name).
    """
    if split == "test":
        return load_trocr_iam_test(start=start, num_lines=num_lines), TROCR_DATASET_NAME
    return load_local_iam_lines(data_dir, split, start=start, num_lines=num_lines), LOCAL_DATASET_NAME


def _slice(items, start, num_lines):
    return items[start:] if num_lines is None else items[start : start + num_lines]


def load_trocr_iam_test(start=0, num_lines=None, root=None):
    """Load the IAM test set released with TrOCR (2,915 lines), mirrored on Kaggle hub.

    Unlike Teklia/IAM-line, these are the line images the TrOCR authors evaluated on (background
    removed, padded) and the transcriptions keep contractions joined ("it's"). The download is
    cached by kagglehub; pass root to use an existing copy (the directory containing IAM/).
    """
    if root is None:
        import kagglehub

        root = kagglehub.dataset_download(TROCR_DATASET_NAME)
    iam_dir = Path(root) / "IAM"
    with open(iam_dir / "gt_test.txt", encoding="utf-8") as f:
        entries = [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]
    entries = _slice(entries, start, num_lines)
    data = {"image": [str(iam_dir / "image" / name) for name, _ in entries], "text": [text for _, text in entries]}
    return Dataset.from_dict(data).cast_column("image", Image())


def load_local_iam_lines(root=DATA_DIR, split="test", start=0, num_lines=None):
    """Load a split of the original IAM line release from root, laid out as extracted archives:

        root/splits/<split>.uttlist   form ids, one per line (Aachen partition)
        root/lines.txt                IAM's line index: id, segmentation, ..., text with "|" between words
        root/lines/<a01>/<form>/<line>.png

    Every line of a form is kept, including those IAM marks "err", which gives the same
    6,482 / 976 / 2,915 lines as Teklia/IAM-line. Words are joined by spaces ("one , but it 's"),
    as in Teklia/IAM-line.
    """
    root = Path(root)
    lines_by_form = {}
    with open(root / "lines.txt", encoding="utf-8") as f:
        for entry in f:
            if entry.startswith("#") or not entry.strip():
                continue
            line_id, *_, text = entry.rstrip("\n").split(" ", 8)
            form = line_id.rsplit("-", 1)[0]
            lines_by_form.setdefault(form, []).append((line_id, text.replace("|", " ")))
    images, texts = [], []
    for form in (root / "splits" / f"{split}.uttlist").read_text().split():
        for line_id, text in lines_by_form[form]:
            images.append(str(root / "lines" / form.split("-")[0] / form / f"{line_id}.png"))
            texts.append(text)
    data = {"image": _slice(images, start, num_lines), "text": _slice(texts, start, num_lines)}
    return Dataset.from_dict(data).cast_column("image", Image())


def load_iam_lines(split="test", start=0, num_lines=None):
    """Load a split of Teklia/IAM-line, optionally sliced to [start, start + num_lines).

    Teklia/IAM-line uses the Aachen partition (6,482 train / 976 validation / 2,915 test lines),
    the same one the TrOCR paper reports on.
    """
    if start or num_lines is not None:
        end = "" if num_lines is None else start + num_lines
        split = f"{split}[{start}:{end}]"
    return load_dataset(DATASET_NAME, split=split)


def iter_batches(dataset, batch_size):
    """Yield (offset, images, texts) batches, with images converted to RGB."""
    for offset in range(0, len(dataset), batch_size):
        batch = dataset[offset : offset + batch_size]
        yield offset, [img.convert("RGB") for img in batch["image"]], batch["text"]


def image_id(split, index):
    """The dataset has no id column, so the row index within the split serves as the image id."""
    return f"{split}_{index:05d}"
