"""
Contract for models that score how well two text spans connect.

Any concrete predictor exposes ``predict(prev, nxt)`` returning a float in
``[0, 1]`` describing the probability that ``nxt`` continues ``prev``.
"""

import abc


# pylint: disable=too-few-public-methods
class ScorerConnector(abc.ABC):
    # Interface every continuity scorer must satisfy.

    @property
    @abc.abstractmethod
    def available(self):  # backing resources loaded and ready for inference
        ...

    @abc.abstractmethod
    def initialize(self):  # load tokenizer and weights into memory
        ...

    @abc.abstractmethod
    def compute_score(self, prev, nxt):  # continuity probability for (prev, nxt)
        ...

    def predict(self, prev, nxt):
        # Convenience alias forwarding to compute_score.
        return self.compute_score(prev, nxt)


class BatchScorer:
    # Adapter that runs a list of span pairs through a single scorer.

    def __init__(self, scorer):
        self._delegate = scorer

    def evaluate_batch(self, pairs):
        collected = []
        for entry in pairs:
            collected.append(self._delegate.compute_score(entry[0], entry[1]))
        return collected


def create_predictor(loader, evaluator):
    # Build a lazily-initialised scorer wrapper.
    # ``loader`` produces the real scorer; ``evaluator`` performs any one-off
    # preparation (e.g. warming weights) once it exists.

    class _Deferred(ScorerConnector):
        def __init__(self):
            self._payload = None

        @property
        def available(self):
            return self._payload is not None and self._payload.available

        def initialize(self):
            if self._payload is None:
                self._payload = loader()
            evaluator(self._payload)

        def compute_score(self, prev, nxt):
            if not self.available:
                raise RuntimeError("predictor has not been initialised")
            return self._payload.compute_score(prev, nxt)

    return _Deferred()


