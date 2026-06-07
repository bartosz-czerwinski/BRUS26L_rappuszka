"""Porównywanie szablonów biometrycznych (embeddingów twarzy).

InsightFace (ArcFace, model buffalo_l) zwraca embedding o wymiarze 512.
Do porównania używamy podobieństwa kosinusowego:
  - zakres [-1, 1], gdzie wyższa wartość = bardziej podobne twarze,
  - ta sama osoba zwykle > ~0.4, różne osoby < ~0.3.

Te funkcje są czyste (tylko numpy), więc działają w testach bez modelu ML.
W architekturze docelowej ten moduł działa po stronie SERWERA.
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
