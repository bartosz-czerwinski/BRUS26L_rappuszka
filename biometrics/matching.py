"""
Porównywanie szablonów biometrycznych (embeddingów twarzy).
"""
from __future__ import annotations

import numpy as np

# Domyślny próg dla podobieństwa kosinusowego embeddingów ArcFace.
DEFAULT_THRESHOLD = 0.4


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Podobieństwo kosinusowe dwóch wektorów. Zwraca wartość w [-1, 1]."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Niezgodne wymiary embeddingów: {a.shape} vs {b.shape}")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_match(a: np.ndarray, b: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Czy dwa szablony należą do tej samej osoby (similarity >= próg)."""
    return cosine_similarity(a, b) >= threshold
