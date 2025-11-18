import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

try:
    import jieba
except ImportError:  # pragma: no cover - dependency installed in production
    jieba = None

from ..database import SessionLocal
from ..models import Command, CommandSettings
from ..utils import to_iso

DEFAULT_MATCH_THRESHOLD = float(os.getenv("COMMAND_MATCH_THRESHOLD", "0.75"))

_BACKEND_SEMANTIC = "semantic"
_BACKEND_BM25_FUZZ = "bm25_fuzz"
_VALID_BACKENDS = {_BACKEND_SEMANTIC, _BACKEND_BM25_FUZZ}

DEFAULT_MATCH_BACKEND = os.getenv("COMMAND_MATCH_BACKEND", _BACKEND_BM25_FUZZ).strip().lower()
if DEFAULT_MATCH_BACKEND not in _VALID_BACKENDS:
    DEFAULT_MATCH_BACKEND = _BACKEND_BM25_FUZZ

BM25_TOP_K = max(1, int(os.getenv("COMMAND_BM25_TOPK", "10")))


def _resolve_backend(choice: Optional[str]) -> str:
    if choice:
        normalized = choice.strip().lower()
        if normalized in _VALID_BACKENDS:
            return normalized
    return DEFAULT_MATCH_BACKEND


COMMAND_STATUS_ENABLED = "enabled"
COMMAND_STATUS_DISABLED = "disabled"
_VALID_COMMAND_STATUSES = {COMMAND_STATUS_ENABLED, COMMAND_STATUS_DISABLED}


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
        if not clean:
            continue
        key = _normalize_for_matching(clean)
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(clean)
    return items


def _normalize_for_matching(text: str) -> str:
    # casefold handles uppercase English without affecting Chinese characters
    return (text or "").strip().casefold()


def _normalize_status(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in _VALID_COMMAND_STATUSES:
        raise ValueError("Invalid command status")
    return normalized


def _tokenize(text: str) -> List[str]:
    content = _normalize_for_matching(text)
    if not content:
        return []
    if jieba is not None:
        return [token.strip() for token in jieba.cut(content, cut_all=False) if token.strip()]
    # fallback to character-level tokens to stay robust on ASR noise
    return [char for char in content if not char.isspace()]


@dataclass
class CommandMatchResult:
    matched: bool
    command: Optional[str]
    score: float


@dataclass
class SemanticMatcherState:
    texts: Tuple[str, ...]
    vectors: np.ndarray


@dataclass
class Bm25MatcherState:
    texts: Tuple[str, ...]
    normalized_texts: Tuple[str, ...]
    bm25: Optional[BM25Okapi]


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
    def __init__(
        self,
        session_factory=SessionLocal,
        *,
        backend: str = DEFAULT_MATCH_BACKEND,
        embedding_service: Optional["CommandEmbeddingService"] = None,
    ):
        self._session_factory = session_factory
        self._backend = backend
        self._embedding_service = embedding_service
        self._cache: Dict[int, Union[SemanticMatcherState, Bm25MatcherState]] = {}
        self._lock = RLock()

    def invalidate(self, user_id: int) -> None:
        with self._lock:
            self._cache.pop(user_id, None)

    def _load_semantic_state(self, rows: Sequence[Command], db: Session) -> SemanticMatcherState:
        texts = tuple(row.text for row in rows)
        if not rows:
            return SemanticMatcherState(texts=texts, vectors=np.empty((0, 0), dtype=np.float32))
        vectors: List[Optional[np.ndarray]] = [None] * len(rows)
        missing_indices: List[int] = []
        for idx, row in enumerate(rows):
            if row.embedding:
                vectors[idx] = np.frombuffer(row.embedding, dtype=np.float32)
            else:
                missing_indices.append(idx)
        if missing_indices:
            if not self._embedding_service:
                raise ValueError("Semantic backend selected but embedding service unavailable")
            missing_texts = [rows[idx].text for idx in missing_indices]
            new_vectors = self._embedding_service.encode(missing_texts)
            for offset, idx in enumerate(missing_indices):
                vector = new_vectors[offset]
                rows[idx].embedding = vector.tobytes()
                db.add(rows[idx])
                vectors[idx] = vector
            db.commit()
        if any(vec is None for vec in vectors):
            raise ValueError("Failed to build semantic matcher state due to missing embeddings")
        matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
        return SemanticMatcherState(texts=texts, vectors=matrix)

    def _load_bm25_state(self, rows: Sequence[Command]) -> Bm25MatcherState:
        texts = tuple(row.text for row in rows)
        normalized_texts = tuple(_normalize_for_matching(text) for text in texts)
        if not rows:
            return Bm25MatcherState(texts=texts, normalized_texts=normalized_texts, bm25=None)
        tokenized = [_tokenize(text) for text in texts]
        model = BM25Okapi(tokenized) if any(tokens for tokens in tokenized) else None
        return Bm25MatcherState(texts=texts, normalized_texts=normalized_texts, bm25=model)

    def _load_from_db(self, user_id: int) -> Union[SemanticMatcherState, Bm25MatcherState]:
        with self._session_factory() as db:
            rows: List[Command] = (
                db.query(Command)
                .filter(Command.user_id == user_id)
                .filter(Command.status == COMMAND_STATUS_ENABLED)
                .order_by(Command.created_at.asc(), Command.id.asc())
                .all()
            )
            if self._backend == _BACKEND_SEMANTIC:
                return self._load_semantic_state(rows, db)
            return self._load_bm25_state(rows)

    def get_state(self, user_id: int) -> Union[SemanticMatcherState, Bm25MatcherState]:
        with self._lock:
            if user_id in self._cache:
                return self._cache[user_id]
            state = self._load_from_db(user_id)
            self._cache[user_id] = state
            return state


class CommandService:
    def __init__(
        self,
        *,
        session_factory=SessionLocal,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self._session_factory = session_factory
        self._backend = _resolve_backend(backend)
        self._embedding: Optional[CommandEmbeddingService] = None
        if self._backend == _BACKEND_SEMANTIC:
            self._embedding = CommandEmbeddingService(model_path=model_path)
        self._matcher = CommandMatcher(
            session_factory=session_factory,
            backend=self._backend,
            embedding_service=self._embedding,
        )

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

    def list_commands(self, user_id: int, *, page: int = 1, page_size: int = 20) -> Dict[str, object]:
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 200))
        with self._get_session() as db:
            base_query = (
                db.query(Command)
                .filter(Command.user_id == user_id)
                .order_by(Command.created_at.asc(), Command.id.asc())
            )
            total = base_query.count()
            commands: List[Command] = (
                base_query.offset((page - 1) * page_size).limit(page_size).all()
            )
        settings = self.get_settings(user_id)

        def _ts(value):
            return to_iso(value) if value else None

        return {
            "enabled": bool(settings.enable_matching),
            "match_threshold": settings.match_threshold or self.default_threshold,
            "items": [
                {
                    "id": cmd.id,
                    "text": cmd.text,
                    "status": cmd.status,
                    "created_at": _ts(cmd.created_at),
                    "updated_at": _ts(cmd.updated_at),
                }
                for cmd in commands
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
            "updated_at": _ts(settings.updated_at),
        }

    def upload_commands(self, user_id: int, commands: Sequence[str]) -> int:
        normalized = _normalize_commands(commands)
        if not normalized:
            raise ValueError("No valid commands provided")
        if self._backend == _BACKEND_SEMANTIC:
            if not self._embedding:
                raise ValueError("Semantic backend requires embedding service")
            vectors: Sequence[Optional[np.ndarray]] = self._embedding.encode(normalized)
        else:
            vectors = [None] * len(normalized)
        with self._get_session() as db:
            for idx, text in enumerate(normalized):
                vector = vectors[idx]
                embedding_bytes = vector.tobytes() if vector is not None else b""
                existing = (
                    db.query(Command)
                    .filter(Command.user_id == user_id, Command.text == text)
                    .one_or_none()
                )
                if existing:
                    existing.embedding = embedding_bytes
                else:
                    db.add(
                        Command(
                            user_id=user_id,
                            text=text,
                            status=COMMAND_STATUS_ENABLED,
                            embedding=embedding_bytes,
                        )
                    )
            db.commit()
        self._matcher.invalidate(user_id)
        return len(normalized)

    def search_commands(
        self,
        user_id: int,
        keyword: str,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, object]:
        query = (keyword or "").strip()
        if not query:
            return {"items": [], "total": 0, "page": 1, "page_size": page_size}
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 200))
        like_pattern = f"%{query}%"
        with self._get_session() as db:
            base_query = (
                db.query(Command)
                .filter(Command.user_id == user_id)
                .filter(Command.text.like(like_pattern))
                .order_by(Command.created_at.asc(), Command.id.asc())
            )
            total = base_query.count()
            rows: List[Command] = (
                base_query.offset((page - 1) * page_size).limit(page_size).all()
            )

        def _ts(value):
            return to_iso(value) if value else None

        return {
            "items": [
                {
                    "id": row.id,
                    "text": row.text,
                    "status": row.status,
                    "created_at": _ts(row.created_at),
                    "updated_at": _ts(row.updated_at),
                }
                for row in rows
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

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

    def delete_command(self, user_id: int, command_id: int) -> bool:
        with self._get_session() as db:
            command = (
                db.query(Command)
                .filter(Command.user_id == user_id, Command.id == command_id)
                .one_or_none()
            )
            if not command:
                return False
            db.delete(command)
            db.commit()
        self._matcher.invalidate(user_id)
        return True

    def update_command(self, user_id: int, command_id: int, text: str) -> Dict[str, object]:
        new_text = (text or "").strip()
        if not new_text:
            raise ValueError("Command text cannot be empty")
        with self._get_session() as db:
            command = (
                db.query(Command)
                .filter(Command.user_id == user_id, Command.id == command_id)
                .one_or_none()
            )
            if not command:
                raise ValueError("Command not found")
            duplicate = (
                db.query(Command)
                .filter(Command.user_id == user_id, Command.text == new_text, Command.id != command_id)
                .first()
            )
            if duplicate:
                raise ValueError("Command text already exists")
            command.text = new_text
            if self._backend == _BACKEND_SEMANTIC:
                if not self._embedding:
                    raise ValueError("Semantic backend requires embedding service")
                vector = self._embedding.encode([new_text])[0]
                command.embedding = vector.tobytes()
            else:
                command.embedding = b""
            db.commit()
            db.refresh(command)
        self._matcher.invalidate(user_id)

        def _ts(value):
            return to_iso(value) if value else None

        return {
            "id": command.id,
            "text": command.text,
            "status": command.status,
            "created_at": _ts(command.created_at),
            "updated_at": _ts(command.updated_at),
        }

    def update_command_status(self, user_id: int, command_id: int, status: str) -> Dict[str, object]:
        new_status = _normalize_status(status)
        with self._get_session() as db:
            command = (
                db.query(Command)
                .filter(Command.user_id == user_id, Command.id == command_id)
                .one_or_none()
            )
            if not command:
                raise ValueError("Command not found")
            command.status = new_status
            db.commit()
            db.refresh(command)
        self._matcher.invalidate(user_id)

        def _ts(value):
            return to_iso(value) if value else None

        return {
            "id": command.id,
            "text": command.text,
            "status": command.status,
            "created_at": _ts(command.created_at),
            "updated_at": _ts(command.updated_at),
        }

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

        state = self._matcher.get_state(user_id)
        if self._backend == _BACKEND_SEMANTIC:
            if not isinstance(state, SemanticMatcherState):
                raise ValueError("Unexpected matcher state for semantic backend")
            return self._match_with_semantic(content, state, threshold)
        if not isinstance(state, Bm25MatcherState):
            raise ValueError("Unexpected matcher state for BM25 backend")
        return self._match_with_bm25(content, state, threshold)

    def _match_with_semantic(
        self,
        content: str,
        state: SemanticMatcherState,
        threshold: float,
    ) -> CommandMatchResult:
        if state.vectors.size == 0 or len(state.texts) == 0:
            return CommandMatchResult(False, None, 0.0)
        if not self._embedding:
            raise ValueError("Semantic backend requires embedding service")
        query_vec = self._embedding.encode([content])[0]
        scores = state.vectors @ query_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < threshold:
            return CommandMatchResult(False, None, best_score)
        return CommandMatchResult(True, state.texts[best_idx], best_score)

    def _match_with_bm25(
        self,
        content: str,
        state: Bm25MatcherState,
        threshold: float,
    ) -> CommandMatchResult:
        if not state.texts or state.bm25 is None:
            return CommandMatchResult(False, None, 0.0)
        normalized_query = _normalize_for_matching(content)
        query_tokens = _tokenize(content)
        if not query_tokens:
            return CommandMatchResult(False, None, 0.0)
        scores = np.asarray(state.bm25.get_scores(query_tokens), dtype=np.float32)
        if scores.size == 0:
            return CommandMatchResult(False, None, 0.0)
        top_k = min(BM25_TOP_K, scores.size)
        sorted_indices = np.argsort(scores)[-top_k:][::-1]
        best_score = 0.0
        best_text: Optional[str] = None
        for idx in sorted_indices:
            idx_int = int(idx)
            candidate_text = state.texts[idx_int]
            candidate_normalized = state.normalized_texts[idx_int]
            fuzzy_score = float(fuzz.token_set_ratio(normalized_query, candidate_normalized))
            if fuzzy_score > best_score:
                best_score = fuzzy_score
                best_text = candidate_text
        normalized = best_score / 100.0
        if not best_text or normalized < threshold:
            return CommandMatchResult(False, None, normalized)
        return CommandMatchResult(True, best_text, normalized)


@lru_cache(maxsize=1)
def get_command_service() -> CommandService:
    return CommandService()


__all__ = [
    "CommandService",
    "CommandMatchResult",
    "get_command_service",
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_MATCH_BACKEND",
]
