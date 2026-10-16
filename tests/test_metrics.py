import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from htr.metrics import NORMALIZATIONS, score  # noqa: E402

join_contractions = NORMALIZATIONS["contractions"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("but it 's a good start .", "but it's a good start ."),
        ("You 're a star", "You're a star"),
        ("I don 't know", "I don't know"),
        ("I 'll go , we 've been", "I'll go , we've been"),
        ("I 'm sure he 'd", "I'm sure he'd"),
        ("IT 'S LATE", "IT'S LATE"),
    ],
)
def test_contractions_are_joined(text, expected):
    assert join_contractions(text) == expected


@pytest.mark.parametrize("text", ["he said ' hello '", "the boys ' toys", "it's already joined", "( I T V , 9.35 p.m. ) ."])
def test_other_text_is_unchanged(text):
    assert join_contractions(text) == text


def test_contraction_spacing_is_not_an_error_after_normalization():
    references, predictions = ["it 's a good start ."], ["it's a good start ."]
    assert score(references, predictions, "none")["cer"] > 0
    assert score(references, predictions, "contractions") == {"cer": 0.0, "wer": 0.0}


def test_score_is_case_sensitive():
    assert score(["Hello"], ["hello"])["cer"] == pytest.approx(1 / 5)
