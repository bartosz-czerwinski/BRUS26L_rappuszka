"""Testy frontendu i endpointów kiosku (wiring, bez prawdziwej twarzy)."""
from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


INVALID_IMAGES = ["to-nie-jest-obraz", "to-nie-jest-obraz", "to-nie-jest-obraz"]


def test_index_is_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Klub Rappuszki" in resp.text


def test_kiosk_identify_rejects_invalid_image():
    # niepoprawne dane obrazu -> 400 (zanim dojdzie do modelu twarzy)
    resp = client.post("/kiosk/identify", json={"images": INVALID_IMAGES})
    assert resp.status_code == 400


def test_kiosk_earn_rejects_invalid_image():
    resp = client.post(
        "/kiosk/earn",
        json={"customer_id": 1, "amount_pln": 10, "images": INVALID_IMAGES},
    )
    assert resp.status_code == 400


def test_kiosk_redeem_rejects_invalid_image():
    resp = client.post(
        "/kiosk/redeem",
        json={"customer_id": 1, "reward_id": "kawa", "images": INVALID_IMAGES},
    )
    assert resp.status_code == 400
