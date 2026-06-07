"""Testy end-to-end programu lojalnościowego (Faza 2).

Używają izolowanej bazy SQLite w pamięci i syntetycznych embeddingów
(nie potrzebują modelu ML ani kamery).
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.database import Base, get_db
from server.loyalty import serialize_embedding, verify_customer
from server.main import app
from server.models import BiometricTemplate, Customer


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


# Syntetyczne embeddingi (zamiast prawdziwych twarzy)
ALICE = [1.0, 0.0, 0.0, 0.0]
ALICE_NOISY = [0.95, 0.05, 0.02, 0.01]   # "ta sama" twarz z lekkim szumem
BOB = [0.0, 1.0, 0.0, 0.0]               # inna osoba (ortogonalny wektor)


def _enroll(client, name, embedding, consent=True):
    return client.post("/enroll", json={"name": name, "consent": consent, "embedding": embedding})


def test_enroll_requires_consent(client):
    resp = _enroll(client, "Alicja", ALICE, consent=False)
    assert resp.status_code == 400


def test_enroll_and_identify_same_person(client):
    enroll = _enroll(client, "Alicja", ALICE)
    assert enroll.status_code == 201
    cid = enroll.json()["customer_id"]

    ident = client.post("/identify", json={"embedding": ALICE_NOISY})
    assert ident.status_code == 200
    body = ident.json()
    assert body["customer_id"] == cid
    assert body["similarity"] >= 0.4


def test_identify_unknown_person(client):
    _enroll(client, "Alicja", ALICE)
    ident = client.post("/identify", json={"embedding": BOB})
    assert ident.status_code == 404


def test_earn_points(client):
    cid = _enroll(client, "Alicja", ALICE).json()["customer_id"]
    resp = client.post("/points/earn", json={"customer_id": cid, "amount_pln": 23.50})
    assert resp.status_code == 200
    body = resp.json()
    assert body["earned"] == 23          # 1 rops / 1 zl, pelne zlotowki
    assert body["points_balance"] == 23


def test_redeem_reward(client):
    cid = _enroll(client, "Alicja", ALICE).json()["customer_id"]
    client.post("/points/earn", json={"customer_id": cid, "amount_pln": 250})  # 250 ropsow

    resp = client.post("/points/redeem", json={"customer_id": cid, "reward_id": "kawa"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["spent"] == 200
    assert body["points_balance"] == 50


def test_redeem_insufficient_points(client):
    cid = _enroll(client, "Alicja", ALICE).json()["customer_id"]
    client.post("/points/earn", json={"customer_id": cid, "amount_pln": 10})  # tylko 10 ropsow
    resp = client.post("/points/redeem", json={"customer_id": cid, "reward_id": "kawa"})
    assert resp.status_code == 400


def test_redeem_unknown_reward(client):
    cid = _enroll(client, "Alicja", ALICE).json()["customer_id"]
    client.post("/points/earn", json={"customer_id": cid, "amount_pln": 500})
    resp = client.post("/points/redeem", json={"customer_id": cid, "reward_id": "samolot"})
    assert resp.status_code == 404


def test_verify_customer_1to1():
    """Weryfikacja 1:1 używana przy zakupie/wymianie punktów."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    customer = Customer(name="Alicja", consent_given=True)
    customer.template = BiometricTemplate(embedding=serialize_embedding(ALICE))
    db.add(customer)
    db.commit()
    db.refresh(customer)

    ok_same, _ = verify_customer(db, customer.id, ALICE_NOISY)
    ok_other, _ = verify_customer(db, customer.id, BOB)
    ok_missing, _ = verify_customer(db, 9999, ALICE)

    assert ok_same is True          # ta sama osoba
    assert ok_other is False        # ktos inny
    assert ok_missing is False      # nieistniejacy klient


def test_delete_customer_rodo(client):
    cid = _enroll(client, "Alicja", ALICE).json()["customer_id"]
    assert client.delete(f"/customer/{cid}").status_code == 204
    # po usunieciu nie da sie zidentyfikowac
    assert client.post("/identify", json={"embedding": ALICE}).status_code == 404
    assert client.get(f"/customer/{cid}").status_code == 404
