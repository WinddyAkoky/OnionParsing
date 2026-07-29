"""
Next-Sentence-Prediction scoring adapter.

Wraps a BERT-style sequence-classification checkpoint to estimate the
probability that one span of text is the natural successor of another.
"""

from __future__ import annotations

import logging
from itertools import chain

import torch

_log = logging.getLogger("onion.reorder.nsp")


def _resolve_transformers():
    """Import the transformers classes lazily, tolerating their absence.

    Returns a ``(tokenizer_cls, model_cls)`` tuple, or ``(None, None)`` when
    the optional ``transformers`` dependency is not installed.
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError:
        return None, None
    return AutoTokenizer, AutoModelForSequenceClassification


_Tokenizer, _Model = _resolve_transformers()

# BERT NSP vocabulary specials kept as constants: [CLS] then [SEP].
_SPECIALS = (101, 102)
# Tokens reserved for the NSP layout: [CLS] A [SEP] B [SEP].
_RESERVED = 3


class NSPPredictor:
    """Scores how plausibly ``second`` continues ``first``."""

    def __init__(self, model_dir, device, maxlen=256):
        (self._src, self._hw, self._budget) = (model_dir, device, maxlen)
        self._net = self._enc = None

    # --------------------------------------------------------------- lifecycle
    @property
    def available(self):
        # Armed once the classification head has been materialised.
        return self._net is not None

    loaded = available

    def _arm(self):
        # Build the tokenizer and classification head onto the target device.
        if _Tokenizer is None or _Model is None:
            raise ImportError("the 'transformers' package is required")
        _log.info("arming NSP model from %s", self._src)
        self._net = _Model.from_pretrained(self._src).to(self._hw).eval()
        self._enc = _Tokenizer.from_pretrained(self._src)
        _log.info("NSP model armed")

    def initialize(self):
        self._arm()

    def prepare(self):
        self._arm()

    def load_model(self):
        self._arm()

    # ----------------------------------------------------------------- scoring
    def compute_score(self, first, second):
        # Probability, in [0, 1], that ``second`` follows ``first``.
        net = self._net
        if net is None:
            raise RuntimeError("NSP model has not been armed")
        inputs = self._materialise(first, second)
        with torch.no_grad():
            probs = net(**inputs).logits.softmax(dim=-1)
        return probs.view(-1)[1].item()

    def evaluate(self, prior, subsequent):
        return self.compute_score(prior, subsequent)

    def predict(self, prior, subsequent):
        return self.compute_score(prior, subsequent)

    # --------------------------------------------------------------- internals
    def _materialise(self, first, second):
        # Build model inputs, trimming only when the budget is exceeded.
        a_ids, b_ids = (
            self._enc.encode(text, add_special_tokens=False)
            for text in (first, second)
        )
        if self._over_budget(a_ids, b_ids):
            return self._fit_batch(a_ids, b_ids)

        cfg = dict(
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self._budget,
        )
        batch = self._enc(first, second, **cfg)
        return dict(batch.to(self._hw))

    def _over_budget(self, a, b):
        # True when ``a`` and ``b`` cannot fit alongside the special tokens.
        return sum(map(len, (a, b))) + _RESERVED > self._budget

    def _fit_batch(self, a, b):
        # Keep the tail of ``a`` and the head of ``b`` within the budget.
        cls, sep = _SPECIALS
        slot = (self._budget - _RESERVED) >> 1
        kept_a = a[len(a) - min(slot, len(a)):]
        kept_b = b[:min(slot, len(b))]

        sequence = list(chain((cls,), kept_a, (sep,), kept_b, (sep,)))
        input_ids = torch.as_tensor([sequence], dtype=torch.long, device=self._hw)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
