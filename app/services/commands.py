import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import Command, CommandSettings

DEFAULT_MATCH_THRESHOLD = float(os.getenv("COMMAND_MATCH_THRESHOLD", "0.75"))


def _default_model_path() -> str:
    env_path = os.getenv("COMMAND_MODEL_PATH")
    if env_path:
        return env_path
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "models" / "bge-small-zh")


def _normalize_commands(commands: Sequence[str]) -> List[str]:
    items = []
    seen = set()
    for text in commands:
        if not text:
            continue
        clean = text.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        items.append(clean)
    return items


@dataclass
class CommandMatchResult:
    matched: bool
    command: Optional[str]
    score: float


class CommandEmbeddingService:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or _default_model_path()
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"command embedding model not found: {self.model_path}")
        self._model = SentenceTransformer(self.model_path)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No commands provided for embedding")
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class CommandMatcher:
    def __init__(self, session_factory=SessionLocal):
        self._session_factory = session_factory
        self._cache: Dict[int, Tuple[np.ndarray, Tuple[str, ...]]] = {}
        self._lock = RLock()

    def invalidate(self, user_id: int) -> None:
        with self._lock:
            self._cache.pop(user_id, None)

    def _load_from_db(self, user_id: int) -> Tuple[np.ndarray, Tuple[str, ...]]:
        with self._session_factory() as db:
            rows: List[Command] = (
                db.query(Command)
                .filter(Command.user_id == user_id)
                .order_by(Command.created_at.asc(), Command.id.asc())
                .all()
            )
        texts = tuple(row.text for row in rows)
        if not rows:
            matrix = np.empty((0, 0), dtype=np.float32)
        else:
            vectors = [np.frombuffer(row.embedding, dtype=np.float32) for row in rows]
            matrix = np.vstack(vectors)
        return matrix, texts

    def get_vectors(self, user_id: int) -> Tuple[np.ndarray, Tuple[str, ...]]:
        with self._lock:
            if user_id in self._cache:
                return self._cache[user_id]
            matrix, texts = self._load_from_db(user_id)
            self._cache[user_id] = (matrix, texts)
            return matrix, texts


class CommandService:
    def __init__(
        self,
        *,
        session_factory=SessionLocal,
        model_path: Optional[str] = None,
    ):
        self._session_factory = session_factory
        self._embedding = CommandEmbeddingService(model_path=model_path)
        self._matcher = CommandMatcher(session_factory=session_factory)

    @property
    def default_threshold(self) -> float:
        return DEFAULT_MATCH_THRESHOLD

    def _get_session(self) -> Session:
        return self._session_factory()

    def get_settings(self, user_id: int) -> CommandSettings:
        with self._get_session() as db:
            settings = (
                db.query(CommandSettings)
                .filter(CommandSettings.user_id == user_id)
                .first()
            )
            if settings:
                return settings
            settings = CommandSettings(
                user_id=user_id,
                enable_matching=False,
                match_threshold=self.default_threshold,
            )
            db.add(settings)
            db.commit()
            db.refresh(settings)
            return settings

    def list_commands(self, user_id: int) -> Dict[str, object]:
        with self._get_session() as db:
            commands: List[Command] = (
                db.query(Command)
                .filter(Command.user_id == user_id)
                .order_by(Command.created_at.asc(), Command.id.asc())
                .all()
            )
        settings = self.get_settings(user_id)

        def _ts(value):
            return value.isoformat() if value else None

        return {
            "enabled": bool(settings.enable_matching),
            "match_threshold": settings.match_threshold or self.default_threshold,
            "commands": [
                {
                    "id": cmd.id,
                    "text": cmd.text,
                    "created_at": _ts(cmd.created_at),
                    "updated_at": _ts(cmd.updated_at),
                }
                for cmd in commands
            ],
            "updated_at": _ts(settings.updated_at),
        }

    def upload_commands(self, user_id: int, commands: Sequence[str]) -> int:
        normalized = _normalize_commands(commands)
        if not normalized:
            raise ValueError("No valid commands provided")
        vectors = self._embedding.encode(normalized)
        with self._get_session() as db:
            for text, vector in zip(normalized, vectors):
                existing = (
                    db.query(Command)
                    .filter(Command.user_id == user_id, Command.text == text)
                    .one_or_none()
                )
                if existing:
                    existing.embedding = vector.tobytes()
                else:
                    db.add(
                        Command(
                            user_id=user_id,
                            text=text,
                            embedding=vector.tobytes(),
                        )
                    )
            db.commit()
        self._matcher.invalidate(user_id)
        return len(normalized)

    def update_matching_state(
        self,
        user_id: int,
        *,
        enabled: bool,
        match_threshold: Optional[float] = None,
    ) -> CommandSettings:
        threshold = match_threshold
        if threshold is not None:
            threshold = max(0.0, min(1.0, float(threshold)))
        with self._get_session() as db:
            settings = (
                db.query(CommandSettings)
                .filter(CommandSettings.user_id == user_id)
                .first()
            )
            if settings is None:
                settings = CommandSettings(user_id=user_id)
                db.add(settings)
            settings.enable_matching = enabled
            if threshold is not None:
                settings.match_threshold = threshold
            elif settings.match_threshold is None:
                settings.match_threshold = self.default_threshold
            db.commit()
            db.refresh(settings)
        return settings

    def match_command(
        self,
        user_id: int,
        text: str,
        *,
        threshold_override: Optional[float] = None,
        settings: Optional[CommandSettings] = None,
    ) -> CommandMatchResult:
        content = (text or "").strip()
        if not content:
            return CommandMatchResult(False, None, 0.0)

        current_settings = settings or self.get_settings(user_id)
        if not current_settings.enable_matching:
            return CommandMatchResult(False, None, 0.0)

        threshold = threshold_override or current_settings.match_threshold or self.default_threshold

        vectors, texts = self._matcher.get_vectors(user_id)
        if vectors.size == 0 or len(texts) == 0:
            return CommandMatchResult(False, None, 0.0)

        query_vec = self._embedding.encode([content])[0]
        scores = vectors @ query_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < threshold:
            return CommandMatchResult(False, None, best_score)
        return CommandMatchResult(True, texts[best_idx], best_score)


@lru_cache(maxsize=1)
def get_command_service() -> CommandService:
    return CommandService()


__all__ = [
    "CommandService",
    "CommandMatchResult",
    "get_command_service",
    "DEFAULT_MATCH_THRESHOLD",
]
