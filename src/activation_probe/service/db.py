"""Postgres logging for every /classify request."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()  # loaded here so DATABASE_URL is correct regardless of import order

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://probe:probe@localhost:5432/activation_probe")

# Deliberately the legacy Column-based declarative style (not SQLAlchemy 2.0's
# Mapped/mapped_column) so this module works unmodified inside Airflow's own
# Python environment, which pins SQLAlchemy 1.4 -- forcing an upgrade there to
# use 2.0-only syntax isn't worth the risk to Airflow's own DB layer.
Base = declarative_base()


class ClassificationLog(Base):
    __tablename__ = "classification_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    flagged = Column(Boolean, nullable=False)
    category = Column(String(64), nullable=True)
    model_layer_used = Column(Integer, nullable=False)
    model_name = Column(String(128), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EvalHistory(Base):
    """One row per nightly (or manual) rerun of the eval harness against a
    held-out split -- lets accuracy/AUC be tracked over time instead of only
    known at training time."""

    __tablename__ = "eval_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    split_name = Column(String(64), nullable=False)
    model_name = Column(String(128), nullable=False)
    probe_layer = Column(Integer, nullable=False)
    n_examples = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=False)
    auc = Column(Float, nullable=True)
    f1 = Column(Float, nullable=False)
    false_positive_rate = Column(Float, nullable=False)
    false_negative_rate = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Alert(Base):
    """A drift alert raised by the nightly DAG. `alert_type` is one of
    'eval_accuracy_drift', 'eval_auc_drift', or 'flag_rate_drift'."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(64), nullable=False)
    message = Column(Text, nullable=False)
    metric_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


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
