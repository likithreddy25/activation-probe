"""Postgres logging for every /classify request."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

load_dotenv()  # loaded here so DATABASE_URL is correct regardless of import order

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://probe:probe@localhost:5432/activation_probe")


class Base(DeclarativeBase):
    pass


class ClassificationLog(Base):
    __tablename__ = "classification_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_layer_used: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def log_classification(
    session: Session,
    prompt: str,
    score: float,
    flagged: bool,
    model_layer_used: int,
    model_name: str,
    category: str | None = None,
) -> ClassificationLog:
    row = ClassificationLog(
        prompt=prompt,
        score=score,
        flagged=flagged,
        category=category,
        model_layer_used=model_layer_used,
        model_name=model_name,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row
