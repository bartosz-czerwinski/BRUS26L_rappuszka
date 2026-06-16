"""
Konfiguracja serwera.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BRUS_", env_file=".env", extra="ignore")

    app_name: str = "BRUS Loyalty Biometrics API"
    version: str = "0.1.0"

    # Próg dopasowania embeddingów twarzy; wyższa wartość oznacza bardziej restrykcyjną weryfikację.
    match_threshold: float = 0.4

    database_url: str = "sqlite:///./brus.db"

    # Model anti-spoofing jest wymagany dla endpointów kioskowych.
    passive_liveness_enabled: bool = True
    passive_liveness_min_frames: int = 3
    passive_liveness_max_frames: int = 6

    # W trybie facenox_logits wynik jest liczony jako sigmoid(real_logit - spoof_logit).
    passive_liveness_threshold: float = 0.6
    anti_spoofing_onnx_path: str = "models/anti_spoofing/best_model_quantized.onnx"
    anti_spoofing_model_img_size: int = 128
    anti_spoofing_bbox_expansion_factor: float = 1.5

    # Dostępne tryby: facenox_logits albo softmax_index.
    anti_spoofing_score_mode: str = "facenox_logits"
    anti_spoofing_live_index: int = 0


settings = Settings()
