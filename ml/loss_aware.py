"""CUBIC that asks the loss classifier before backing off."""
from functools import lru_cache

import joblib
import numpy as np

from baselines.cubic import Cubic

from .features import FlowMonitor
from .loss_classifier import MODEL_PATH


@lru_cache(maxsize=1)
def load_classifier():
    return joblib.load(MODEL_PATH)["model"]


class LossAwareCubic(Cubic):
    """Standard CUBIC, except that losses the classifier judges to be random are ignored."""

    name = "CUBIC+ML"

    def __init__(self, classifier=None, threshold: float = 0.2):
        super().__init__()
        self.classifier = classifier or load_classifier()
        # Back off unless the classifier is at least 80% sure the loss was random: wrongly
        # ignoring a congestion loss takes bandwidth from competing flows, so err towards backing off.
        self.threshold = threshold
        self.monitor = FlowMonitor()
        self.ignored = 0

    def on_ack(self, ack):
        self.monitor.on_ack(ack)
        super().on_ack(ack)

    def on_loss(self, loss):
        features = self.monitor.features(loss.now)
        self.monitor.on_loss(loss)
        if loss.send_time < self.recovery_start:
            return
        if self.classifier.predict_proba(np.array([features]))[0, 1] < self.threshold:
            self.ignored += 1
            return
        super().on_loss(loss)
