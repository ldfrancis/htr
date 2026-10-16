"""Text normalizations and corpus-level CER/WER.

CER and WER are corpus-level (total edits / total reference characters or words), computed
with jiwer on case-sensitive text. Each normalization is applied to both the reference and the
prediction:

    none           raw text, as in the TrOCR paper
    contractions   IAM splits contractions ("it 's") but TrOCR writes them joined ("it's"),
                   so " 's", " 't", " 'll", ... are joined to the preceding word
"""

import re

import jiwer

CONTRACTION = re.compile(r" '(s|t|d|m|ll|re|ve)\b", re.IGNORECASE)

NORMALIZATIONS = {
    "none": lambda text: text,
    "contractions": lambda text: CONTRACTION.sub(r"'\1", text),
}


def score(references, predictions, normalization="none"):
    """Return {"cer": ..., "wer": ...} after applying the named normalization to both sides."""
    normalize = NORMALIZATIONS[normalization]
    refs = [normalize(text) for text in references]
    preds = [normalize(text) for text in predictions]
    return {"cer": jiwer.cer(refs, preds), "wer": jiwer.wer(refs, preds)}
