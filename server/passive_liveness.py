"""Sprawdzanie twarzy / face anti-spoofing dla endpointów kiosku.

Ta wersja NIE używa już heurystyk OpenCV jako fallbacku. Poprzednia heurystyka
potrafiła przepuszczać zwykłe statyczne zdjęcie, bo z pojedynczego obrazu RGB
nie da się wiarygodnie wywnioskować żywotności prostymi regułami.

Obecnie backend działa w trybie fail-closed:
- jeśli model ONNX anti-spoofing istnieje, uruchamia model,
- jeśli modelu nie ma, odrzuca próbę i podaje instrukcję konfiguracji.

Rekomendowany lekki model do projektu:
facenox/face-antispoof-onnx -> models/best_model_quantized.onnx
Zakładany format wyjścia tego modelu: dwa logity [real, spoof].
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from biometrics.engine import analyze
from biometrics.matching import cosine_similarity
from server.config import settings


@dataclass(frozen=True)
class PassiveFrame:
    """Dane jednej automatycznie pobranej klatki."""
    image_bgr: np.ndarray
    face_crop_rgb: np.ndarray
    embedding: list[float]
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class AntiSpoofingResult:
    """Wynik kontroli anti-spoofing."""
    is_live: bool
    score: float
    reason: str


def embedding_after_passive_liveness_check(images: list[str]) -> list[float]:
    """Sprawdza twarz i zwraca embedding środkowej klatki.

    Funkcja jest używana przez endpointy /kiosk/*. Zwykłe endpointy embeddingowe
    /enroll i /identify pozostają bez zmian.
    """
    if not images:
        raise ValueError("Nie przesłano żadnej klatki z kamery.")

    if not settings.passive_liveness_enabled:
        frame = _extract_single_face(images[0], index=1)
        return frame.embedding

    _validate_frame_count(images)
    frames = [_extract_single_face(image_data, index=i + 1) for i, image_data in enumerate(images)]
    _require_same_person(frames)

    result = _run_anti_spoofing(frames)
    if not result.is_live:
        raise ValueError("Wykryto zdjęcie. Użyj prawdziwej twarzy przed kamerą.")

    return frames[len(frames) // 2].embedding


def _validate_frame_count(images: list[str]) -> None:
    if len(images) < settings.passive_liveness_min_frames:
        raise ValueError(
            f"Do sprawdzenia twarzy wymagane są co najmniej "
            f"{settings.passive_liveness_min_frames} klatki."
        )
    if len(images) > settings.passive_liveness_max_frames:
        raise ValueError(
            f"Do sprawdzenia twarzy można przesłać maksymalnie "
            f"{settings.passive_liveness_max_frames} klatek."
        )


def _extract_single_face(image_data: str, index: int) -> PassiveFrame:
    image_bgr = _decode_image(image_data)
    faces = analyze(image_bgr)

    if len(faces) == 0:
        raise ValueError(f"Klatka {index}: nie wykryto twarzy.")
    if len(faces) > 1:
        raise ValueError(f"Klatka {index}: wykryto {len(faces)} twarze — oczekiwano jednej.")

    face = faces[0]
    bbox = tuple(float(x) for x in face.bbox)
    face_crop_rgb = _crop_face_for_antispoofing(image_bgr, bbox)
    embedding = face.normed_embedding.tolist()
    return PassiveFrame(image_bgr=image_bgr, face_crop_rgb=face_crop_rgb, embedding=embedding, bbox=bbox)


def _decode_image(image_data: str) -> np.ndarray:
    """Dekoduje obraz z data URL/base64 do BGR OpenCV."""
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data)
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError("Nieprawidłowy obraz.") from exc

    if image is None:
        raise ValueError("Nie udało się zdekodować obrazu.")
    return image


def _crop_face_for_antispoofing(image_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Wycina kwadratowy crop twarzy z rozszerzeniem, zgodnie z typowym MiniFAS preprocessingiem."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    face_w = max(x2 - x1, 1.0)
    face_h = max(y2 - y1, 1.0)
    max_dim = max(face_w, face_h)
    scale = max(float(settings.anti_spoofing_bbox_expansion_factor), 1.0)
    crop_size = int(max_dim * scale)

    center_x = x1 + face_w / 2.0
    center_y = y1 + face_h / 2.0
    left = int(center_x - crop_size / 2.0)
    top = int(center_y - crop_size / 2.0)
    right = left + crop_size
    bottom = top + crop_size

    crop_left = max(left, 0)
    crop_top = max(top, 0)
    crop_right = min(right, width)
    crop_bottom = min(bottom, height)

    crop = image_rgb[crop_top:crop_bottom, crop_left:crop_right]
    if crop.size == 0:
        raise ValueError("Nie udało się wyciąć twarzy z obrazu.")

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
    return crop


def _require_same_person(frames: list[PassiveFrame]) -> None:
    """Chroni przed dosłaniem kilku klatek różnych osób."""
    first = frames[0].embedding
    for frame in frames[1:]:
        score = cosine_similarity(first, frame.embedding)
        if score < settings.match_threshold:
            raise ValueError("Klatki nie wyglądają na tę samą osobę.")


def _run_anti_spoofing(frames: list[PassiveFrame]) -> AntiSpoofingResult:
    """Uruchamia model ONNX. Brak modelu = odrzucenie próby."""
    model_path = Path(settings.anti_spoofing_onnx_path)
    if not model_path.is_file():
        raise ValueError(
            "Brak modelu anti-spoofing ONNX. Pobierz model poleceniem: "
            "python tools/download_antispoof_model.py albo ustaw "
            "BRUS_ANTI_SPOOFING_ONNX_PATH na poprawną ścieżkę. "
            "Celowo nie używam już heurystycznego fallbacku, bo przepuszczał zdjęcia."
        )
    return _onnx_anti_spoofing(frames, model_path)


@lru_cache(maxsize=1)
def _load_onnx_session(model_path: str):
    import onnxruntime as ort

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _onnx_anti_spoofing(frames: list[PassiveFrame], model_path: Path) -> AntiSpoofingResult:
    """Wariant modelowy dla MiniFAS/ONNX."""
    session = _load_onnx_session(str(model_path))
    input_name = session.get_inputs()[0].name
    batch = _preprocess_batch([frame.face_crop_rgb for frame in frames], settings.anti_spoofing_model_img_size)
    output = session.run(None, {input_name: batch})[0]
    logits = np.asarray(output, dtype=np.float32)

    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    if logits.shape[0] != len(frames):
        raise ValueError(
            "Nieprawidłowy kształt wyjścia modelu anti-spoofing: "
            f"oczekiwano batch={len(frames)}, otrzymano {logits.shape}."
        )

    if settings.anti_spoofing_score_mode == "facenox_logits":
        scores = [_score_facenox_logits(row) for row in logits]
        reason = "model ONNX: sigmoid(real_logit - spoof_logit)"
    elif settings.anti_spoofing_score_mode == "softmax_index":
        scores = [_score_softmax_index(row, settings.anti_spoofing_live_index) for row in logits]
        reason = "model ONNX: softmax klasy live"
    else:
        raise ValueError(f"Nieznany tryb scoringu anti-spoofing: {settings.anti_spoofing_score_mode}")

    score = float(np.mean(scores))
    return AntiSpoofingResult(
        is_live=score >= settings.passive_liveness_threshold,
        score=score,
        reason=reason,
    )


def _preprocess_batch(face_crops_rgb: list[np.ndarray], model_img_size: int) -> np.ndarray:
    """Letterbox + normalizacja [0,1] + NCHW, zgodnie z typowym MiniFAS ONNX."""
    if not face_crops_rgb:
        raise ValueError("Brak cropów twarzy dla modelu anti-spoofing.")

    batch = np.zeros((len(face_crops_rgb), 3, model_img_size, model_img_size), dtype=np.float32)
    for i, crop in enumerate(face_crops_rgb):
        batch[i] = _preprocess_crop(crop, model_img_size)
    return batch


def _preprocess_crop(crop_rgb: np.ndarray, model_img_size: int) -> np.ndarray:
    old_h, old_w = crop_rgb.shape[:2]
    if old_h <= 0 or old_w <= 0:
        raise ValueError("Pusty crop twarzy.")

    ratio = float(model_img_size) / max(old_h, old_w)
    new_h = max(1, int(old_h * ratio))
    new_w = max(1, int(old_w * ratio))
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=interpolation)

    delta_w = model_img_size - new_w
    delta_h = model_img_size - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    chw = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw


def _score_facenox_logits(logits: np.ndarray) -> float:
    """Dla facenox/face-antispoof-onnx: output [real_logit, spoof_logit]."""
    if logits.size < 2:
        raise ValueError("Model anti-spoofing powinien zwracać co najmniej 2 wartości.")
    real_logit = float(logits[0])
    spoof_logit = float(logits[1])
    return _sigmoid(real_logit - spoof_logit)


def _score_softmax_index(logits: np.ndarray, live_index: int) -> float:
    probs = _softmax(logits.astype(np.float32).reshape(-1))
    if live_index < 0 or live_index >= probs.size:
        raise ValueError("Nieprawidłowy indeks klasy live dla modelu anti-spoofing.")
    return float(probs[live_index])


def _sigmoid(value: float) -> float:
    value = max(min(value, 60.0), -60.0)
    return float(1.0 / (1.0 + np.exp(-value)))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)
