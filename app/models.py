from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, func

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


class AdminAccount(Base):
    __tablename__ = "admin_accounts"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    role = Column(String, nullable=False, default="admin")
    is_active = Column(Boolean, default=True)
    is_builtin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    last_login_at = Column(DateTime(timezone=True), nullable=True, index=True)


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


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    username = Column(String, nullable=True)
    dominant_speaker = Column(String, nullable=True)
    speakers = Column(Text, nullable=True)
    text = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    similarity_avg = Column(Float, nullable=True)
    similarity_max = Column(Float, nullable=True)
    segments_count = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="in_progress")
    locale = Column(String, nullable=True)
    channel = Column(String, nullable=True)
    operator = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        index=True,
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True, index=True)
    transcript_id = Column(
        Integer,
        ForeignKey("transcripts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_id = Column(Integer, nullable=False)
    speaker_name = Column(String, nullable=True)
    speaker_user_id = Column(Integer, nullable=True)
    similarity = Column(Float, nullable=True)
    start_ms = Column(Integer, nullable=True)
    end_ms = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    topk = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


__all__ = ["User", "AdminAccount", "EventLog", "Transcript", "TranscriptSegment"]
