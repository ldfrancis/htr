"""TrOCR text-line recognizer."""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-base-handwritten"
MODEL_NAME_MAP = {
    "trocr-base": "microsoft/trocr-base-handwritten",
    "trocr-large": "microsoft/trocr-large-handwritten",
    "trocr-small": "microsoft/trocr-small-handwritten",
}


class Recognizer:
    """Transcribes batches of text-line images with a TrOCR checkpoint."""

    def __init__(self, model_name=MODEL_NAME, device=None):
        """model_name is a key of MODEL_NAME_MAP (e.g. "trocr-large") or a Hugging Face model id."""
        self.model_name = MODEL_NAME_MAP.get(model_name, model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images, num_beams=1, max_new_tokens=128):
        """Return one transcription per RGB image. num_beams=1 is greedy decoding."""
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
