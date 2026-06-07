"""
Konfiguracja serwera.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BRUS_", env_file=".env", extra="ignore")

    app_name: str = "BRUS Loyalty Biometrics API"
    version: str = "0.1.0"

    # Próg dopasowania szablonów (podobieństwo kosinusowe ArcFace, zakres [-1,1]).
    # Wyższy = bardziej restrykcyjnie. Ta sama osoba zwykle > ~0.4.
    match_threshold: float = 0.4

    # Ścieżka bazy danych (Faza 2).
    database_url: str = "sqlite:///./brus.db"

    # Pasywne liveness / anti-spoofing dla endpointów /kiosk/*.
    # UWAGA: od tej wersji nie ma heurystycznego fallbacku. Jeśli modelu ONNX nie ma,
    # backend odrzuca próbę, zamiast fałszywie akceptować zdjęcia.
    passive_liveness_enabled: bool = True
    passive_liveness_min_frames: int = 3
    passive_liveness_max_frames: int = 6

    # Dla modelu facenox/face-antispoof-onnx wynik traktujemy jako logity [real, spoof].
    # score = sigmoid(real_logit - spoof_logit). Próg 0.75 jest dość restrykcyjny.
    passive_liveness_threshold: float = 0.6
    anti_spoofing_onnx_path: str = "models/anti_spoofing/best_model_quantized.onnx"
    anti_spoofing_model_img_size: int = 128
    anti_spoofing_bbox_expansion_factor: float = 1.5

    # Tryby:
    # - "facenox_logits": output modelu [real_logit, spoof_logit], rekomendowane dla facenox.
    # - "softmax_index": klasyczne prawdopodobieństwa/logity, użyj anti_spoofing_live_index.
    anti_spoofing_score_mode: str = "facenox_logits"
    anti_spoofing_live_index: int = 0


settings = Settings()
