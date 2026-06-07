"""
Porównuje przysłany szablon z szablonami wszystkich klientów w bazie
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from biometrics.matching import cosine_similarity
from server.config import settings
from server.models import BiometricTemplate, Customer


def serialize_embedding(embedding: list[float]) -> str:
    return json.dumps(embedding)


def deserialize_embedding(raw: str) -> list[float]:
    return json.loads(raw)


def verify_customer(db: Session, customer_id: int, embedding: list[float]) -> tuple[bool, float]:
    """
    Weryfikacja 1:1 - czy przysłany szablon należy do wskazanego klienta.
    """
    customer = db.get(Customer, customer_id)
    if customer is None or customer.template is None:
        return False, -1.0
    score = cosine_similarity(embedding, deserialize_embedding(customer.template.embedding))
    return score >= settings.match_threshold, score


def find_best_match(db: Session, embedding: list[float]) -> tuple[Customer, float] | None:
    """Zwraca (klient, podobieństwo) dla najlepszego dopasowania powyżej progu,
    albo None gdy nikt nie pasuje."""
    best_customer: Customer | None = None
    best_score = -1.0

    for template in db.query(BiometricTemplate).all():
        score = cosine_similarity(embedding, deserialize_embedding(template.embedding))
        if score > best_score:
            best_score = score
            best_customer = template.customer

    if best_customer is None or best_score < settings.match_threshold:
        return None
    return best_customer, best_score
