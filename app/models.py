from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, func

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    account = Column(String, unique=True, index=True)
    identity = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    status = Column(String, nullable=False, default="enabled")
    embedding = Column(Text, nullable=True)


class EventLog(Base):
    __tablename__ = "event_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    username = Column(String, nullable=True)
    operator = Column(String, nullable=True)
    type = Column(String, nullable=False)
    category = Column(String, nullable=True)
    authorized = Column(Boolean, default=True)
    payload = Column(Text, nullable=True)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )


__all__ = ["User", "EventLog"]
