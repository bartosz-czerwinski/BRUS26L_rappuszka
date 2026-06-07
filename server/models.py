"""Modele ORM: klient, szablon biometryczny, historia transakcji punktowych."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    # Zgoda na przetwarzanie danych biometrycznych (RODO art. 9) — wymagana.
    consent_given: Mapped[bool] = mapped_column(default=False)
    consent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    points_balance: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    template: Mapped[BiometricTemplate] = relationship(
        back_populates="customer", uselist=False, cascade="all, delete-orphan"
    )
    transactions: Mapped[list[PointsTransaction]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )


class BiometricTemplate(Base):
    __tablename__ = "biometric_templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    # Embedding zapisany jako JSON (lista floatów). W Fazie 2 BEZ szyfrowania —
    # szyfrowanie at-rest dodajemy świadomie w Fazie 4 (porównanie "przed/po").
    embedding: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    customer: Mapped[Customer] = relationship(back_populates="template")


class PointsTransaction(Base):
    __tablename__ = "points_transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    kind: Mapped[str] = mapped_column(String(20))  # "earn" | "redeem"
    points: Mapped[int] = mapped_column(Integer)    # dodatnie = naliczenie, ujemne = wydanie
    description: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    customer: Mapped[Customer] = relationship(back_populates="transactions")
