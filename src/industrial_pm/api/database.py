from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from industrial_pm.config import get_settings


class Base(DeclarativeBase):
    pass


class PredictionAudit(Base):
    __tablename__ = "prediction_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    unit_id: Mapped[int] = mapped_column(Integer, index=True)
    model_run_id: Mapped[str] = mapped_column(String(128), index=True)
    predicted_rul: Mapped[float] = mapped_column(Float)
    risk_30: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class RetrainingAudit(Base):
    __tablename__ = "retraining_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def create_tables() -> None:
    Base.metadata.create_all(engine)


async def get_session() -> AsyncGenerator[Session, None]:
    with SessionLocal() as session:
        yield session
