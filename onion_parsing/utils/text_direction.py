"""
Text direction detection using word-frequency likelihood.

Decides whether a line should be read left-to-right or right-to-left by
comparing the jieba log-likelihood of the token stream against its mirrored
counterpart, then flipping the offending lines while mirroring bracket glyphs.
"""

import math
import urllib.request
from pathlib import Path

import jieba

try:
    from onion_parsing.core.logging import get_logger
    _logger = get_logger(__name__)
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)


_DICT_URL = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big"


class TextDirectionJudge:
    """Picks reading orientation from jieba token-frequency statistics."""

    _PAIRS = ("()", "[]", "{}", "<>", "（）", "《》", "【】", "「」")
    BRACKET_MAP = {
        glyph: pair[1 - idx]
        for pair in _PAIRS
        for idx, glyph in enumerate(pair)
    }
    _TRANSLATE = str.maketrans(
        {ord(op): cl for op, cl in BRACKET_MAP.items()}
    )

    def __init__(self):
        self._freqs = {}
        self._corpus_total = 0
        self._load_lexicon()

    # ----------------------------------------------------------- dictionary
    def _load_lexicon(self):
        """Point jieba at the packaged dictionaries, if present."""
        root = Path(__file__).resolve().parents[2]
        primary = root / "resources" / "dict.txt.big"
        if not primary.is_file():
            self._download_dict(primary)
        if primary.is_file():
            jieba.set_dictionary(str(primary))
            jieba.dt.initialize()
        else:
            _logger.warning("Primary dictionary not found: %s", primary)

        user = root / "resources" / "user.txt"
        if user.is_file():
            jieba.load_userdict(str(user))
        else:
            _logger.info("User dictionary not found: %s", user)

        tokenizer = jieba.dt
        self._freqs = tokenizer.FREQ
        self._corpus_total = tokenizer.total

    @staticmethod
    def _download_dict(target: Path) -> None:
        """Download the jieba large dictionary if missing."""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _logger.info("Downloading dictionary to %s ...", target)
            urllib.request.urlretrieve(_DICT_URL, str(target))
            _logger.info("Dictionary downloaded successfully")
        except Exception as exc:
            _logger.warning("Failed to download dictionary: %s", exc)

    # --------------------------------------------------------------- scoring
    def _loglikelihood(self, text):
        """Summed log-probability of the tokenised text under the lexicon."""
        tokens = jieba.lcut(text, HMM=True)
        if not tokens:
            return -100.0

        denominator = float(self._corpus_total)
        score = 0.0
        for piece in tokens:
            count = self._freqs.get(piece)
            if count is None or count < 0.01:
                count = 0.01
            score += math.log(count / denominator)
        return score

    @staticmethod
    def natural_reverse(s):
        """Reverse ``s`` while mirroring bracket pairs (trailing space kept)."""
        cutoff = len(s)
        while cutoff > 0 and s[cutoff - 1].isspace():
            cutoff -= 1
        body = s[:cutoff]
        tail = s[cutoff:]
        return body.translate(TextDirectionJudge._TRANSLATE)[::-1] + tail

    # --------------------------------------------------------------- verdicts
    def judge(self, text, reversed_text):
        """Return ``"LTR"`` or ``"RTL"`` for the more plausible orientation."""
        if len(text) < 2:
            return "LTR"
        if self._loglikelihood(text) < self._loglikelihood(reversed_text):
            return "RTL"
        return "LTR"

    def fix_text(self, text):
        """Mirror every line whose reversed form scores higher."""
        try:
            corrected = []
            for ln in text.split("\n"):
                twin = self.natural_reverse(ln)
                corrected.append(twin if self.judge(ln, twin) == "RTL" else ln)
            joined = "\n".join(corrected)
            return joined
        except Exception:
            return text

    def correct(self, text):
        """Alias for :meth:`fix_text` for backward compatibility."""
        return self.fix_text(text)


judge = TextDirectionJudge()
