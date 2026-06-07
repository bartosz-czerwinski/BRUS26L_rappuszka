"""Konfiguracja bazy danych (SQLAlchemy). SQLite na potrzeby PoC."""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from server.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # wymagane dla SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Zależność FastAPI: sesja bazy na czas obsługi żądania."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
