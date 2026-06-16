"""
Serwer API
"""
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from server import rewards
from server.config import settings
from server.database import Base, engine, get_db
from server.loyalty import find_best_match, serialize_embedding, verify_customer
from server.passive_liveness import embedding_after_passive_liveness_check
from server.models import BiometricTemplate, Customer, PointsTransaction
from server.schemas import (
    CustomerResponse,
    EarnRequest,
    EarnResponse,
    EnrollRequest,
    IdentifyRequest,
    IdentifyResponse,
    KioskEarnRequest,
    KioskEnrollRequest,
    KioskIdentifyRequest,
    KioskRedeemRequest,
    RedeemRequest,
    RedeemResponse,
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name, version=settings.version)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Frontend webowy (Klub Rappuszki)."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/logo.svg", include_in_schema=False)
def logo() -> FileResponse:
    """Logo Rappki."""
    return FileResponse(FRONTEND_DIR / "logo.svg", media_type="image/svg+xml")


@app.get("/health")
def health() -> dict:
    """Sprawdzenie stanu serwera (liveness probe)."""
    return {"status": "ok", "version": settings.version}


@app.get("/rewards")
def list_rewards() -> dict:
    """Katalog nagród do wymiany za punkty."""
    return {
        "brand": rewards.BRAND,
        "points_name": rewards.POINTS_NAME,
        "points_per_pln": rewards.POINTS_PER_PLN,
        "rewards": rewards.REWARDS,
    }


@app.post("/enroll", response_model=CustomerResponse, status_code=201)
def enroll(req: EnrollRequest, db: Session = Depends(get_db)) -> CustomerResponse:
    """Rejestracja klienta w programie wraz z szablonem biometrycznym.

    Wymaga wyraźnej zgody (RODO art. 9) — bez niej rejestracja jest odrzucana.
    """
    if not req.consent:
        raise HTTPException(status_code=400, detail="Wymagana zgoda na przetwarzanie danych biometrycznych.")

    customer = Customer(
        name=req.name,
        consent_given=True,
        consent_at=datetime.now(timezone.utc),
        points_balance=0,
    )
    customer.template = BiometricTemplate(embedding=serialize_embedding(req.embedding))
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return CustomerResponse(customer_id=customer.id, name=customer.name, points_balance=customer.points_balance)


@app.post("/identify", response_model=IdentifyResponse)
def identify(req: IdentifyRequest, db: Session = Depends(get_db)) -> IdentifyResponse:
    """Identyfikacja klienta na podstawie przysłanego szablonu twarzy."""
    match = find_best_match(db, req.embedding)
    if match is None:
        raise HTTPException(status_code=404, detail="Nie rozpoznano klienta.")
    customer, score = match
    return IdentifyResponse(
        customer_id=customer.id,
        name=customer.name,
        points_balance=customer.points_balance,
        similarity=round(score, 4),
    )


@app.post("/kiosk/enroll", response_model=CustomerResponse, status_code=201)
def kiosk_enroll(req: KioskEnrollRequest, db: Session = Depends(get_db)) -> CustomerResponse:
    """Rejestracja z frontendu: sprawdzenie twarzy, embedding, potem enroll."""
    embedding = _embedding_from_passive_liveness(req.images)
    return enroll(EnrollRequest(name=req.name, consent=req.consent, embedding=embedding), db)


@app.post("/kiosk/identify", response_model=CustomerResponse)
def kiosk_identify(req: KioskIdentifyRequest, db: Session = Depends(get_db)) -> CustomerResponse:
    """Identyfikacja z frontendu: sprawdzenie twarzy i zwrócenie konta bez wyniku podobieństwa."""
    embedding = _embedding_from_passive_liveness(req.images)
    match = find_best_match(db, embedding)
    if match is None:
        raise HTTPException(status_code=404, detail="Nie rozpoznano klienta.")
    customer, _score = match
    return CustomerResponse(
        customer_id=customer.id,
        name=customer.name,
        points_balance=customer.points_balance,
    )


@app.post("/kiosk/earn", response_model=EarnResponse)
def kiosk_earn(req: KioskEarnRequest, db: Session = Depends(get_db)) -> EarnResponse:
    """Naliczenie punktów z frontendu — wymaga sprawdzenia i weryfikacji twarzy."""
    embedding = _embedding_from_passive_liveness(req.images)
    _require_face_verification(db, req.customer_id, embedding)
    return earn_points(EarnRequest(customer_id=req.customer_id, amount_pln=req.amount_pln), db)


@app.post("/kiosk/redeem", response_model=RedeemResponse)
def kiosk_redeem(req: KioskRedeemRequest, db: Session = Depends(get_db)) -> RedeemResponse:
    """Wymiana punktów z frontendu — wymaga sprawdzenia i weryfikacji twarzy."""
    embedding = _embedding_from_passive_liveness(req.images)
    _require_face_verification(db, req.customer_id, embedding)
    return redeem_points(RedeemRequest(customer_id=req.customer_id, reward_id=req.reward_id), db)


@app.post("/points/earn", response_model=EarnResponse)
def earn_points(req: EarnRequest, db: Session = Depends(get_db)) -> EarnResponse:
    """Naliczenie rappsów za zakup o podanej wartości."""
    customer = _get_customer(db, req.customer_id)
    earned = rewards.points_for_purchase(req.amount_pln)
    customer.points_balance += earned
    db.add(PointsTransaction(
        customer_id=customer.id, kind="earn", points=earned,
        description=f"Zakup za {req.amount_pln:.2f} zl",
    ))
    db.commit()
    db.refresh(customer)
    return EarnResponse(
        customer_id=customer.id, name=customer.name,
        points_balance=customer.points_balance, earned=earned,
    )


@app.post("/points/redeem", response_model=RedeemResponse)
def redeem_points(req: RedeemRequest, db: Session = Depends(get_db)) -> RedeemResponse:
    """Wymiana rappsów na nagrodę z katalogu."""
    customer = _get_customer(db, req.customer_id)
    reward = rewards.REWARDS.get(req.reward_id)
    if reward is None:
        raise HTTPException(status_code=404, detail="Nie ma takiej nagrody.")
    if customer.points_balance < reward["cost"]:
        raise HTTPException(
            status_code=400,
            detail=f"Za mało punktów: masz {customer.points_balance}, potrzeba {reward['cost']}.",
        )
    customer.points_balance -= reward["cost"]
    db.add(PointsTransaction(
        customer_id=customer.id, kind="redeem", points=-reward["cost"],
        description=f"Nagroda: {reward['name']}",
    ))
    db.commit()
    db.refresh(customer)
    return RedeemResponse(
        customer_id=customer.id, name=customer.name,
        points_balance=customer.points_balance,
        reward=reward["name"], spent=reward["cost"],
    )


@app.get("/customer/{customer_id}", response_model=CustomerResponse)
def get_customer(customer_id: int, db: Session = Depends(get_db)) -> CustomerResponse:
    """Stan konta klienta."""
    customer = _get_customer(db, customer_id)
    return CustomerResponse(customer_id=customer.id, name=customer.name, points_balance=customer.points_balance)


@app.delete("/customer/{customer_id}", status_code=204)
def delete_customer(customer_id: int, db: Session = Depends(get_db)) -> None:
    """Usunięcie klienta i jego danych biometrycznych (RODO — prawo do usunięcia)."""
    customer = _get_customer(db, customer_id)
    db.delete(customer)
    db.commit()


# Funkcje pomocnicze używane przez endpointy.

def _embedding_from_passive_liveness(images: list[str]) -> list[float]:
    """Sprawdza twarz i zwraca embedding twarzy."""
    try:
        return embedding_after_passive_liveness_check(images)
    except ValueError as exc:
        message = str(exc)
        if "Nieprawidłowy obraz" in message or "zdekodować" in message:
            raise HTTPException(status_code=400, detail=message)
        raise HTTPException(status_code=422, detail=_public_face_check_message(message))


def _public_face_check_message(message: str) -> str:
    """Zamienia techniczne komunikaty modułu sprawdzania twarzy na tekst dla użytkownika."""
    if "Wykryto zdjęcie" in message or "Podejrzenie zdjęcia" in message:
        return "Wykryto zdjęcie. Użyj prawdziwej twarzy przed kamerą."
    if "Brak modelu" in message:
        return "Moduł sprawdzania twarzy nie jest skonfigurowany."
    if "liveness" in message.lower() or "anti-spoofing" in message.lower():
        return "Nie udało się sprawdzić twarzy."
    return message


def _embedding_from_image(image_data: str) -> list[float]:
    """Dekoduje obraz (data URL lub base64) i wyciąga embedding pojedynczej twarzy."""
    import base64

    import cv2
    import numpy as np

    from biometrics.engine import extract_single_embedding

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data)
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Nieprawidłowy obraz.")
    if image is None:
        raise HTTPException(status_code=400, detail="Nie udało się zdekodować obrazu.")
    try:
        return extract_single_embedding(image).tolist()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _require_face_verification(db: Session, customer_id: int, embedding: list[float]) -> None:
    """Sprawdza, że twarz na zdjęciu należy do wskazanego klienta. Inaczej 403."""
    _get_customer(db, customer_id)
    ok, _score = verify_customer(db, customer_id, embedding)
    if not ok:
        raise HTTPException(
            status_code=403,
            detail="Weryfikacja twarzy nieudana.",
        )


def _get_customer(db: Session, customer_id: int) -> Customer:
    customer = db.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Nie ma takiego klienta.")
    return customer
