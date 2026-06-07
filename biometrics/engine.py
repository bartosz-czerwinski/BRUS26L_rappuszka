"""
Ekstrakcja szablonów biometrycznych z obrazu (InsightFace / ArcFace).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _get_app():
    """Leniwie inicjalizuje i cache'uje analizator twarzy (CPU)."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def analyze(image_bgr: np.ndarray) -> list:
    """Zwraca listę wykrytych twarzy (obiekty InsightFace z polami `bbox`
    i `normed_embedding`). Przydatne, gdy potrzebna jest też pozycja twarzy
    w kadrze (np. rysowanie ramki w demie kamery)."""
    return _get_app().get(image_bgr)


def extract_embeddings(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Zwraca listę embeddingów (po jednym na wykrytą twarz) z obrazu BGR.

    Pusta lista = nie wykryto żadnej twarzy.
    """
    return [face.normed_embedding for face in analyze(image_bgr)]


def extract_single_embedding(image_bgr: np.ndarray) -> np.ndarray:
    """Zwraca embedding pojedynczej twarzy z obrazu.

    Wymaga dokładnie jednej twarzy — przy zerze lub wielu rzuca ValueError.
    Taki wymóg jest zamierzony: przy rejestracji/identyfikacji chcemy
    jednoznacznie jednej osoby w kadrze.
    """
    embeddings = extract_embeddings(image_bgr)
    if len(embeddings) == 0:
        raise ValueError("Nie wykryto twarzy na obrazie.")
    if len(embeddings) > 1:
        raise ValueError(f"Wykryto {len(embeddings)} twarzy — oczekiwano jednej.")
    return embeddings[0]


def load_image(path: str) -> np.ndarray:
    """Wczytuje obraz z pliku jako tablicę BGR (format OpenCV)."""
    import cv2

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Nie udało się wczytać obrazu: {path}")
    return image
