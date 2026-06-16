"""
Reguły programu lojalnościowego
"""

BRAND = "Rappka"
POINTS_NAME = "rappsy"
POINTS_PER_PLN = 1


# Katalog nagród: id -> nazwa i koszt w rappsach
REWARDS: dict[str, dict] = {
    "kawa":       {"name": "Kawa z ekspresu", "cost": 200},
    "hotdog":     {"name": "Hot-dog klasyczny", "cost": 150},
    "drozdzowka": {"name": "Drożdżówka z serem", "cost": 120},
    "napoj":      {"name": "Napój 0,5 L", "cost": 100},
}


def points_for_purchase(amount_pln: float) -> int:
    """Liczba rappsów za zakup o danej wartości (pełne złotówki)."""
    return int(amount_pln) * POINTS_PER_PLN
