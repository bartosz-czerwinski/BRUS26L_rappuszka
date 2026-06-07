"""Schematy żądań/odpowiedzi API (Pydantic)."""
from pydantic import BaseModel, Field


class EnrollRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    consent: bool = Field(description="Zgoda na przetwarzanie danych biometrycznych (wymagana)")
    embedding: list[float] = Field(min_length=1)


class IdentifyRequest(BaseModel):
    embedding: list[float] = Field(min_length=1)


class KioskEnrollRequest(BaseModel):
    """Rejestracja z frontendu webowego.

    Frontend przesyła kilka automatycznie pobranych klatek z kamery. Backend
    wykonuje sprawdzenie twarzy i dopiero potem zapisuje embedding.
    """
    name: str = Field(min_length=1, max_length=120)
    consent: bool
    images: list[str] = Field(min_length=3, max_length=6)


class KioskIdentifyRequest(BaseModel):
    """Identyfikacja ze sprawdzeniem twarzy."""
    images: list[str] = Field(min_length=3, max_length=6)


class KioskEarnRequest(BaseModel):
    """Naliczenie punktów ze sprawdzeniem twarzy i weryfikacją właściciela konta."""
    customer_id: int
    amount_pln: float = Field(gt=0)
    images: list[str] = Field(min_length=3, max_length=6)


class KioskRedeemRequest(BaseModel):
    """Wymiana punktów ze sprawdzeniem twarzy i weryfikacją właściciela konta."""
    customer_id: int
    reward_id: str
    images: list[str] = Field(min_length=3, max_length=6)


class EarnRequest(BaseModel):
    customer_id: int
    amount_pln: float = Field(gt=0, description="Wartość zakupu w złotówkach")


class RedeemRequest(BaseModel):
    customer_id: int
    reward_id: str


class CustomerResponse(BaseModel):
    customer_id: int
    name: str
    points_balance: int


class IdentifyResponse(CustomerResponse):
    similarity: float


class TransactionResponse(BaseModel):
    kind: str
    points: int
    description: str


class EarnResponse(CustomerResponse):
    earned: int


class RedeemResponse(CustomerResponse):
    reward: str
    spent: int
