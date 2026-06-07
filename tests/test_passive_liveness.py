"""Testy pasywnego liveness bez uruchamiania InsightFace ani prawdziwego ONNX."""
from types import SimpleNamespace

import numpy as np
import pytest

from server import passive_liveness


VALID_FAKE_IMAGES = ["ZmFrZV9pbWFnZQ==", "ZmFrZV9pbWFnZQ==", "ZmFrZV9pbWFnZQ=="]
EMBEDDING = np.array([1.0, 0.0, 0.0, 0.0])


def test_passive_liveness_accepts_when_model_accepts(monkeypatch):
    def fake_decode(_image_data):
        return np.zeros((140, 140, 3), dtype=np.uint8)

    def fake_analyze(_image):
        return [SimpleNamespace(bbox=np.array([10.0, 10.0, 120.0, 120.0]), normed_embedding=EMBEDDING)]

    def fake_anti_spoofing(_frames):
        return passive_liveness.AntiSpoofingResult(is_live=True, score=0.91, reason="test")

    monkeypatch.setattr(passive_liveness, "_decode_image", fake_decode)
    monkeypatch.setattr(passive_liveness, "analyze", fake_analyze)
    monkeypatch.setattr(passive_liveness, "_run_anti_spoofing", fake_anti_spoofing)

    result = passive_liveness.embedding_after_passive_liveness_check(VALID_FAKE_IMAGES)

    assert result == EMBEDDING.tolist()


def test_passive_liveness_rejects_when_model_rejects(monkeypatch):
    def fake_decode(_image_data):
        return np.zeros((140, 140, 3), dtype=np.uint8)

    def fake_analyze(_image):
        return [SimpleNamespace(bbox=np.array([10.0, 10.0, 120.0, 120.0]), normed_embedding=EMBEDDING)]

    def fake_anti_spoofing(_frames):
        return passive_liveness.AntiSpoofingResult(is_live=False, score=0.12, reason="test")

    monkeypatch.setattr(passive_liveness, "_decode_image", fake_decode)
    monkeypatch.setattr(passive_liveness, "analyze", fake_analyze)
    monkeypatch.setattr(passive_liveness, "_run_anti_spoofing", fake_anti_spoofing)

    with pytest.raises(ValueError, match="Wykryto zdjęcie"):
        passive_liveness.embedding_after_passive_liveness_check(VALID_FAKE_IMAGES)


def test_missing_model_is_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(passive_liveness.settings, "anti_spoofing_onnx_path", str(tmp_path / "missing.onnx"))
    frames = [
        passive_liveness.PassiveFrame(
            image_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
            face_crop_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            embedding=EMBEDDING.tolist(),
            bbox=(0.0, 0.0, 32.0, 32.0),
        )
        for _ in range(3)
    ]

    with pytest.raises(ValueError, match="Brak modelu anti-spoofing ONNX"):
        passive_liveness._run_anti_spoofing(frames)


def test_facenox_logits_scoring():
    live = passive_liveness._score_facenox_logits(np.array([4.0, -2.0], dtype=np.float32))
    spoof = passive_liveness._score_facenox_logits(np.array([-2.0, 4.0], dtype=np.float32))

    assert live > 0.99
    assert spoof < 0.01


def test_preprocess_batch_shape_and_range():
    crops = [np.full((80, 120, 3), 128, dtype=np.uint8) for _ in range(3)]

    batch = passive_liveness._preprocess_batch(crops, model_img_size=128)

    assert batch.shape == (3, 3, 128, 128)
    assert batch.dtype == np.float32
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0
