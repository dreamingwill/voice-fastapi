import os
from typing import Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from .config import (
    ADMIN_DISPLAY_NAME,
    ADMIN_PASSWORD,
    ADMIN_ROLE,
    ADMIN_USERNAME,
    DATABASE_URL,
)
from .utils import hash_password


_DATABASE_URL = DATABASE_URL


def _build_engine(db_url: str):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args)


engine = _build_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    init_engine()
    from . import models  # noqa: F401  Ensure models are registered

    Base.metadata.create_all(bind=engine)
    _ensure_user_columns()
    _ensure_command_code_column()
    _ensure_admin_account()
    _ensure_system_settings()


def init_engine(db_url: Optional[str] = None) -> None:
    global engine, SessionLocal, _DATABASE_URL

    db_url = db_url or DATABASE_URL
    if not db_url or db_url == _DATABASE_URL:
        return
    _DATABASE_URL = db_url
    engine = _build_engine(db_url)
    SessionLocal.configure(bind=engine)
    _ensure_sqlite_file(engine)


def _ensure_sqlite_file(active_engine) -> None:
    try:
        url = active_engine.url
    except Exception:
        return
    if url.drivername != "sqlite":
        return
    path = url.database
    if not path or path == ":memory:":
        return
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    if not os.path.exists(abs_path):
        with open(abs_path, "a", encoding="utf-8"):
            pass


def _ensure_user_columns() -> None:
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    with engine.begin() as conn:
        if "account" not in existing_columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN account VARCHAR(255)"))
        if "phone" not in existing_columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN phone VARCHAR(32)"))
        if "status" not in existing_columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN status VARCHAR(32) DEFAULT 'enabled'"))
        conn.execute(text("UPDATE users SET status = 'enabled' WHERE status IS NULL"))
        conn.execute(text("UPDATE users SET account = username WHERE account IS NULL AND username IS NOT NULL"))

    inspector = inspect(engine)
    indexes = {idx["name"] for idx in inspector.get_indexes("users")}
    if "ix_users_account" not in indexes:
        with engine.begin() as conn:
            conn.execute(text("CREATE UNIQUE INDEX ix_users_account ON users (account)"))


def _ensure_command_code_column() -> None:
    inspector = inspect(engine)
    if "commands" not in inspector.get_table_names():
        return
    existing_columns = {col["name"] for col in inspector.get_columns("commands")}
    if "code" in existing_columns:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE commands ADD COLUMN code VARCHAR(64)"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_admin_account() -> None:
    from .models import AdminAccount

    with SessionLocal() as db:
        super_admin = (
            db.query(AdminAccount)
                .filter(AdminAccount.username == ADMIN_USERNAME)
                .first()
        )
        if super_admin is None:
            db.add(
                AdminAccount(
                    username=ADMIN_USERNAME,
                    password_hash=hash_password(ADMIN_PASSWORD),
                    role=ADMIN_ROLE or "super_admin",
                    display_name=ADMIN_DISPLAY_NAME,
                    is_active=True,
                    is_builtin=True,
                )
            )
            db.commit()


def _ensure_system_settings() -> None:
    from .models import SystemSettings

    with SessionLocal() as db:
        settings = db.query(SystemSettings).order_by(SystemSettings.id).first()
        if settings is None:
            db.add(SystemSettings(enable_speaker_recognition=True))
            db.commit()


__all__ = ["engine", "SessionLocal", "Base", "init_db", "init_engine", "get_db"]
