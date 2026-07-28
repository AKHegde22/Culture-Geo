"""Tests for activation extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.hooks import ActivationExtractor, ActivationCache


def test_activation_cache_init():
    """Verify ActivationCache initialization."""
    cache = ActivationCache()
    assert cache.num_layers == 0
    assert cache.hidden_states == {}


def test_activation_cache_numpy():
    """Verify ActivationCache numpy conversion."""
    import numpy as np
    cache = ActivationCache()
    cache.hidden_states = {0: np.zeros((1, 100))}
    cache.to_numpy()
    assert isinstance(cache.hidden_states[0], np.ndarray)
