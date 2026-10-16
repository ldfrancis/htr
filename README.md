# htr


## Setup

```bash
uv venv && uv pip install -r requirements.txt
```

## Data

By default (`htr.data.load_split`), train and validation come from the original IAM release and
test from the TrOCR authors' release of the IAM test set
([changheonkim/iam-trocr](https://www.kaggle.com/datasets/changheonkim/iam-trocr), downloaded by kagglehub).
The IAM files are read from `../data` (override with `--data-dir`), extracted as:

```
data/
  lines.txt                    IAM's line index and transcriptions
  lines/<a01>/<form>/<line>.png
  splits/{train,validation,test}.uttlist   Aachen partition (form ids)
```

## Usage

```bash
# Transcribe 32 train lines, saving images and output/results.csv
python scripts/recognize.py --split train --num-lines 32

# Evaluate on the full test split (TrOCR release) with the TrOCR paper's beam size
python scripts/evaluate.py --split test --num-beams 10 --predictions-csv output/predictions_trocr_test_beam10.csv

# Evaluate on the original IAM validation lines
python scripts/evaluate.py --split validation --predictions-csv output/predictions_local_validation.csv

# Pick one source for any split: local (original IAM), trocr (test only) or teklia (Teklia/IAM-line)
python scripts/evaluate.py --dataset teklia --split test --num-beams 10 --predictions-csv output/predictions_test_beam10.csv

# Run tests
python -m pytest tests
```

The TrOCR paper reports 3.42% CER for TrOCR-Base, which is reproduced on
[changheonkim/iam-trocr](https://www.kaggle.com/datasets/changheonkim/iam-trocr) (`--dataset trocr`),
a mirror of the IAM test set released with TrOCR. Its transcriptions already join contractions,
so both normalizations give the same score.