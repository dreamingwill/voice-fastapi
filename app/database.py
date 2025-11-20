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


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    from . import models  # noqa: F401  Ensure models are registered

    Base.metadata.create_all(bind=engine)
    _ensure_user_columns()
    _ensure_command_code_column()
    _ensure_admin_account()
    _ensure_system_settings()


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


__all__ = ["engine", "SessionLocal", "Base", "init_db", "get_db"]
