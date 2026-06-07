"""Branding i reguły programu lojalnościowego "Rappka".

Inspirowane polskimi sklepami typu convenience (punkty za zakupy + nagrody do
wymiany), ale marka, nazwy i nagrody są fikcyjne i autorskie — na potrzeby projektu.
"""

BRAND = "Rappka"
POINTS_NAME = "rappsy"       # odpowiednik punktów lojalnościowych
POINTS_PER_PLN = 1           # 1 rapps za każdą pełną złotówkę zakupów


# Katalog nagród: id -> (nazwa, koszt w ropsach)
REWARDS: dict[str, dict] = {
    "kawa":       {"name": "Kawa z ekspresu", "cost": 200},
    "hotdog":     {"name": "Hot-dog klasyczny", "cost": 150},
    "drozdzowka": {"name": "Drożdżówka z serem", "cost": 120},
    "napoj":      {"name": "Napój 0,5 L", "cost": 100},
}


def points_for_purchase(amount_pln: float) -> int:
    """Liczba rappsów za zakup o danej wartości (pełne złotówki)."""
    return int(amount_pln) * POINTS_PER_PLN
