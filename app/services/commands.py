import os
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sqlalchemy.orm import Session

try:
    import jieba
except ImportError:  # pragma: no cover - dependency installed in production
    jieba = None

from ..database import SessionLocal
from ..models import Command, CommandSettings
from ..utils import to_iso

DEFAULT_MATCH_THRESHOLD = float(os.getenv("COMMAND_MATCH_THRESHOLD", "0.75"))

BM25_TOP_K = max(1, int(os.getenv("COMMAND_BM25_TOPK", "10")))


COMMAND_STATUS_ENABLED = "enabled"
COMMAND_STATUS_DISABLED = "disabled"
_VALID_COMMAND_STATUSES = {COMMAND_STATUS_ENABLED, COMMAND_STATUS_DISABLED}


def _normalize_code(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    code = value.strip()
    return code or None


def _normalize_commands(commands: Sequence["CommandCreatePayload"]) -> List["CommandCreatePayload"]:
    items: List[CommandCreatePayload] = []
    seen_texts = set()
    seen_codes = set()
    for command in commands:
        text = command.text or ""
        clean = text.strip()
        if not clean:
            continue
        key = _normalize_for_matching(clean)
        if not key or key in seen_texts:
            continue
        seen_texts.add(key)
        normalized_code = _normalize_code(command.code)
        if normalized_code:
            if normalized_code in seen_codes:
                raise ValueError("Duplicate command code in upload payload")
            seen_codes.add(normalized_code)
        items.append(CommandCreatePayload(text=clean, code=normalized_code))
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
class CommandCreatePayload:
    text: str
    code: Optional[str] = None


@dataclass
class CommandMatchResult:
    matched: bool
    command: Optional[str]
    score: float
    command_id: Optional[int] = None
    command_code: Optional[str] = None


@dataclass
class Bm25MatcherState:
    texts: Tuple[str, ...]
    normalized_texts: Tuple[str, ...]
    command_ids: Tuple[int, ...]
    command_codes: Tuple[Optional[str], ...]
    bm25: Optional[BM25Okapi]


class CommandMatcher:
    def __init__(
        self,
        session_factory=SessionLocal,
    ):
        self._session_factory = session_factory
        self._cache: Dict[int, Bm25MatcherState] = {}
        self._lock = RLock()

    def invalidate(self, user_id: int) -> None:
        with self._lock:
            self._cache.pop(user_id, None)

    def _load_bm25_state(self, rows: Sequence[Command]) -> Bm25MatcherState:
        texts = tuple(row.text for row in rows)
        normalized_texts = tuple(_normalize_for_matching(text) for text in texts)
        command_ids = tuple(int(row.id) for row in rows)
        command_codes = tuple(row.code for row in rows)
        if not rows:
            return Bm25MatcherState(
                texts=texts,
                normalized_texts=normalized_texts,
                command_ids=command_ids,
                command_codes=command_codes,
                bm25=None,
            )
        tokenized = [_tokenize(text) for text in texts]
        model = BM25Okapi(tokenized) if any(tokens for tokens in tokenized) else None
        return Bm25MatcherState(
            texts=texts,
            normalized_texts=normalized_texts,
            command_ids=command_ids,
            command_codes=command_codes,
            bm25=model,
        )

    def _load_from_db(self, user_id: int) -> Bm25MatcherState:
        with self._session_factory() as db:
            rows: List[Command] = (
                db.query(Command)
                .filter(Command.user_id == user_id)
                .filter(Command.status == COMMAND_STATUS_ENABLED)
                .order_by(Command.created_at.asc(), Command.id.asc())
                .all()
            )
            return self._load_bm25_state(rows)

    def get_state(self, user_id: int) -> Bm25MatcherState:
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
    ):
        self._session_factory = session_factory
        self._matcher = CommandMatcher(session_factory=session_factory)

    @property
    def default_threshold(self) -> float:
        return DEFAULT_MATCH_THRESHOLD

    def _get_session(self) -> Session:
        return self._session_factory()

    def _ensure_unique_code(
        self,
        db: Session,
        user_id: int,
        code: Optional[str],
        *,
        exclude_command_id: Optional[int] = None,
    ) -> None:
        if not code:
            return
        query = (
            db.query(Command)
            .filter(Command.user_id == user_id)
            .filter(Command.code == code)
        )
        if exclude_command_id is not None:
            query = query.filter(Command.id != exclude_command_id)
        exists = query.first()
        if exists:
            raise ValueError("Command code already exists")

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
                    "code": cmd.code,
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

    def upload_commands(self, user_id: int, commands: Sequence[CommandCreatePayload]) -> int:
        normalized = _normalize_commands(commands)
        if not normalized:
            raise ValueError("No valid commands provided")
        with self._get_session() as db:
            for payload in normalized:
                existing = (
                    db.query(Command)
                    .filter(Command.user_id == user_id, Command.text == payload.text)
                    .one_or_none()
                )
                if payload.code:
                    self._ensure_unique_code(
                        db,
                        user_id,
                        payload.code,
                        exclude_command_id=existing.id if existing else None,
                    )
                if existing:
                    existing.embedding = b""
                    if payload.code is not None:
                        existing.code = payload.code
                else:
                    db.add(
                        Command(
                            user_id=user_id,
                            text=payload.text,
                            code=payload.code,
                            status=COMMAND_STATUS_ENABLED,
                            embedding=b"",
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
                    "code": row.code,
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

    def update_command(
        self,
        user_id: int,
        command_id: int,
        text: str,
        *,
        code: Optional[str] = None,
        update_code: bool = False,
    ) -> Dict[str, object]:
        new_text = (text or "").strip()
        if not new_text:
            raise ValueError("Command text cannot be empty")
        normalized_code = _normalize_code(code) if update_code else None
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
            if update_code:
                self._ensure_unique_code(db, user_id, normalized_code, exclude_command_id=command.id)
                command.code = normalized_code
            command.text = new_text
            command.embedding = b""
            db.commit()
            db.refresh(command)
        self._matcher.invalidate(user_id)

        def _ts(value):
            return to_iso(value) if value else None

        return {
            "id": command.id,
            "text": command.text,
            "code": command.code,
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
            "code": command.code,
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
        if not isinstance(state, Bm25MatcherState):
            raise ValueError("Unexpected matcher state for BM25 backend")
        return self._match_with_bm25(content, state, threshold)

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
        best_code: Optional[str] = None
        best_id: Optional[int] = None
        for idx in sorted_indices:
            idx_int = int(idx)
            candidate_text = state.texts[idx_int]
            candidate_normalized = state.normalized_texts[idx_int]
            fuzzy_score = float(fuzz.token_set_ratio(normalized_query, candidate_normalized))
            if fuzzy_score > best_score:
                best_score = fuzzy_score
                best_text = candidate_text
                best_code = state.command_codes[idx_int] if idx_int < len(state.command_codes) else None
                best_id = state.command_ids[idx_int] if idx_int < len(state.command_ids) else None
        normalized = best_score / 100.0
        if not best_text or normalized < threshold:
            return CommandMatchResult(False, None, normalized)
        return CommandMatchResult(True, best_text, normalized, command_id=best_id, command_code=best_code)


@lru_cache(maxsize=1)
def get_command_service() -> CommandService:
    return CommandService()


__all__ = [
    "CommandService",
    "CommandMatchResult",
    "CommandCreatePayload",
    "get_command_service",
    "DEFAULT_MATCH_THRESHOLD",
]
