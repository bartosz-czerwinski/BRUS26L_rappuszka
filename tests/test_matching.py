"""Testy logiki dopasowania szablonów (bez modelu ML — syntetyczne wektory)."""
import numpy as np
import pytest

from biometrics.matching import cosine_similarity, is_match


def test_identical_vectors_are_match():
    v = np.array([0.1, 0.2, 0.3, 0.4])
    assert cosine_similarity(v, v) == pytest.approx(1.0)
    assert is_match(v, v)


def test_orthogonal_vectors_not_match():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)
    assert not is_match(a, b)


def test_opposite_vectors_not_match():
    a = np.array([1.0, 1.0])
    b = np.array([-1.0, -1.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)
    assert not is_match(a, b)


def test_similar_vectors_match_above_threshold():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.9, 0.1, 0.05])
    assert is_match(a, b, threshold=0.4)


def test_mismatched_dimensions_raise():
    with pytest.raises(ValueError):
        cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_zero_vector_returns_zero():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, b) == 0.0
