import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sqlalchemy.orm import Session
from sqlalchemy import case

try:
    import jieba
except ImportError:  # pragma: no cover - dependency installed in production
    jieba = None
try:
    from pypinyin import Style, pinyin
except ImportError:  # pragma: no cover - dependency installed in production
    pinyin = None
    Style = None

from ..database import SessionLocal
from ..models import Command, CommandSettings
from ..utils import to_iso

DEFAULT_MATCH_THRESHOLD = float(os.getenv("COMMAND_MATCH_THRESHOLD", "0.75"))
# Use a fixed global user id for storing and matching commands/settings without binding to any admin account
GLOBAL_USER_ID = int(os.getenv("GLOBAL_COMMAND_USER_ID", "0"))

BM25_TOP_K = max(1, int(os.getenv("COMMAND_BM25_TOPK", "10")))
PHONETIC_ENABLED = os.getenv("COMMAND_PHONETIC_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
PHONETIC_WEIGHT = float(os.getenv("COMMAND_PHONETIC_WEIGHT", "0.4"))
PHONETIC_MIN_TEXT_SCORE = float(os.getenv("COMMAND_PHONETIC_MIN_TEXT", "0.4"))
PHONETIC_THRESHOLD = float(os.getenv("COMMAND_PHONETIC_THRESHOLD", "0.68"))
PHONETIC_TONE = os.getenv("COMMAND_PHONETIC_TONE", "false").strip().lower() in {"1", "true", "yes", "on"}
NUMERIC_MISMATCH_PENALTY = float(os.getenv("COMMAND_NUMERIC_MISMATCH_PENALTY", "0.4"))
NUMERIC_MISSING_PENALTY = float(os.getenv("COMMAND_NUMERIC_MISSING_PENALTY", "0.6"))
DEFAULT_COMMAND_TEXT_ALIASES_PATH = (
    Path(os.getenv("COMMAND_TEXT_ALIASES_PATH", ""))
    if os.getenv("COMMAND_TEXT_ALIASES_PATH")
    else Path(__file__).resolve().parents[2] / "config" / "command_text_aliases.json"
)


COMMAND_STATUS_ENABLED = "enabled"
COMMAND_STATUS_DISABLED = "disabled"
_VALID_COMMAND_STATUSES = {COMMAND_STATUS_ENABLED, COMMAND_STATUS_DISABLED}

logger = logging.getLogger("command.match")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _normalize_code(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    code = value.strip()
    return code or None


def _normalize_commands(commands: Sequence["CommandCreatePayload"]) -> List["CommandCreatePayload"]:
    items: List[CommandCreatePayload] = []
    seen_texts = set()
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


def _normalize_alias_mapping(data: Any) -> Dict[str, str]:
    if not isinstance(data, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key, value in data.items():
        src = str(key or "").strip()
        dst = str(value or "").strip()
        if src and dst and src != dst:
            normalized[src] = dst
    return normalized


@lru_cache(maxsize=1)
def _load_command_text_aliases() -> Dict[str, Dict[str, str]]:
    path = DEFAULT_COMMAND_TEXT_ALIASES_PATH
    if not path.is_file():
        return {"exact_aliases": {}, "replacements": {}}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning("command.aliases load_failed path=%s error=%s", path, exc)
        return {"exact_aliases": {}, "replacements": {}}
    if not isinstance(data, dict):
        return {"exact_aliases": {}, "replacements": {}}
    return {
        "exact_aliases": _normalize_alias_mapping(data.get("exact_aliases")),
        "replacements": _normalize_alias_mapping(data.get("replacements")),
    }


def normalize_command_text(text: str) -> str:
    content = (text or "").strip()
    if not content:
        return ""
    aliases = _load_command_text_aliases()
    exact_aliases = aliases.get("exact_aliases", {})
    if content in exact_aliases:
        normalized = exact_aliases[content]
        logger.info("command.normalize type=exact original=%s normalized=%s", content, normalized)
        return normalized
    normalized = content
    replacements = aliases.get("replacements", {})
    for src, dst in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if src in normalized:
            normalized = normalized.replace(src, dst)
    normalized = normalized.strip()
    if normalized != content:
        logger.info("command.normalize type=replace original=%s normalized=%s", content, normalized)
    return normalized


def _tokenize(text: str) -> List[str]:
    content = _normalize_for_matching(text)
    if not content:
        return []
    if jieba is not None:
        return [token.strip() for token in jieba.cut(content, cut_all=False) if token.strip()]
    # fallback to character-level tokens to stay robust on ASR noise
    return [char for char in content if not char.isspace()]


def _build_pinyin(text: str) -> str:
    if not text:
        return ""
    if pinyin is None:
        return ""
    style = Style.TONE3 if PHONETIC_TONE else Style.NORMAL
    tokens = []
    for item in pinyin(text, style=style, strict=False, errors="ignore"):
        if not item:
            continue
        token = (item[0] or "").strip().casefold()
        if token:
            tokens.append(token)
    return " ".join(tokens)


def _pinyin_token_count(text: str) -> int:
    if not text:
        return 0
    return len([token for token in text.split() if token])


def _pinyin_length_ratio(a: str, b: str) -> float:
    count_a = _pinyin_token_count(a)
    count_b = _pinyin_token_count(b)
    if not count_a or not count_b:
        return 0.0
    return min(count_a, count_b) / max(count_a, count_b)


_CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _parse_chinese_numeral(token: str) -> Optional[int]:
    if not token:
        return None
    if token == "十":
        return 10
    if "十" in token:
        parts = token.split("十", 1)
        tens_token = parts[0]
        ones_token = parts[1] if len(parts) > 1 else ""
        tens = _CHINESE_DIGITS.get(tens_token, 1 if tens_token == "" else None)
        ones = _CHINESE_DIGITS.get(ones_token, 0 if ones_token == "" else None)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if token in _CHINESE_DIGITS:
        return _CHINESE_DIGITS[token]
    return None


def _parse_numeric_token(token: str) -> Optional[int]:
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return _parse_chinese_numeral(token)


def _extract_numeric_tokens(text: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if not text:
        return (), ()
    ordinals: List[int] = []
    numbers: List[int] = []
    for match in re.finditer(r"第([零一二三四五六七八九十两0-9]+)", text):
        value = _parse_numeric_token(match.group(1))
        if value is not None:
            ordinals.append(value)
    for match in re.finditer(r"([零一二三四五六七八九十两0-9]+)(次|遍|轮)", text):
        value = _parse_numeric_token(match.group(1))
        if value is not None:
            ordinals.append(value)
    for match in re.finditer(r"\d+", text):
        numbers.append(int(match.group(0)))
    for match in re.finditer(r"[零一二三四五六七八九十两]+", text):
        value = _parse_chinese_numeral(match.group(0))
        if value is not None:
            numbers.append(value)
    ordinals_sorted = tuple(sorted(set(ordinals)))
    numbers_sorted = tuple(sorted(set(numbers)))
    return ordinals_sorted, numbers_sorted


def _numeric_factor(
    query_ordinals: Tuple[int, ...],
    query_numbers: Tuple[int, ...],
    candidate_ordinals: Tuple[int, ...],
    candidate_numbers: Tuple[int, ...],
) -> float:
    if query_ordinals or candidate_ordinals:
        if query_ordinals and candidate_ordinals:
            if query_ordinals != candidate_ordinals:
                return NUMERIC_MISMATCH_PENALTY
        else:
            return NUMERIC_MISSING_PENALTY
    if query_numbers or candidate_numbers:
        if query_numbers and candidate_numbers:
            if query_numbers != candidate_numbers:
                return NUMERIC_MISMATCH_PENALTY
        else:
            return NUMERIC_MISSING_PENALTY
    return 1.0


@dataclass(frozen=True)
class StageParseResult:
    stage: str = ""
    invalid_timed_stage: bool = False
    has_timed_stage_phrase: bool = False


@dataclass
class CommandCreatePayload:
    text: str
    code: Optional[str] = None


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

@dataclass
class CommandMatchResult:
    matched: bool
    command: Optional[str]
    score: float
    command_id: Optional[int] = None
    command_code: Optional[str] = None
    original_text: str = ""
    normalized_text: str = ""
    intent_detected: bool = False
    match_type: Optional[Literal["exact", "rule", "fuzzy"]] = None


class IntentClassifier:
    """
    Rule-based intent classifier to determine if a text is likely a command.
    """
    MIN_COMMAND_LENGTH = 2
    MAX_COMMAND_LENGTH = 40

    # Keywords that strongly indicate a command intent
    INTENT_KEYWORDS = {
        "检查", "准备", "停", "起飞", "点火", "发射", "各号", "注意", "下发", "程序", "跟踪", "复查"
    }

    # Regex patterns for common command structures
    INTENT_PATTERNS = [
        re.compile(r"^各号.*"),
        re.compile(r".*?准备$"),
        re.compile(r".*?检查$"),
        re.compile(r"^[点火|起飞|发射|停]$"),
    ]

    @classmethod
    def is_intent(cls, text: str) -> bool:
        if not text:
            return False
        
        # Length constraint: commands are usually short
        if len(text) < cls.MIN_COMMAND_LENGTH or len(text) > cls.MAX_COMMAND_LENGTH:
            return False

        # Keyword trigger
        if any(keyword in text for keyword in cls.INTENT_KEYWORDS):
            return True

        # Regex pattern trigger
        if any(pattern.match(text) for pattern in cls.INTENT_PATTERNS):
            return True

        return False


@dataclass
class Bm25MatcherState:
    texts: Tuple[str, ...]
    normalized_texts: Tuple[str, ...]
    pinyin_texts: Tuple[str, ...]
    command_ids: Tuple[int, ...]
    command_codes: Tuple[Optional[str], ...]
    bm25: Optional[BM25Okapi]


class CommandMatcher:
    def __init__(
        self,
        session_factory=SessionLocal,
    ):
        self._session_factory = session_factory
        # Single global cache for command state
        self._cache: Optional[Bm25MatcherState] = None
        self._lock = RLock()

    def invalidate(self, user_id: int) -> None:
        with self._lock:
            # ignore user_id; operate on global cache
            self._cache = None

    def _load_bm25_state(self, rows: Sequence[Command]) -> Bm25MatcherState:
        texts = tuple(row.text for row in rows)
        normalized_texts = tuple(_normalize_for_matching(text) for text in texts)
        pinyin_texts = tuple(_build_pinyin(text) for text in texts)
        command_ids = tuple(int(row.id) for row in rows)
        command_codes = tuple(row.code for row in rows)
        if not rows:
            return Bm25MatcherState(
                texts=texts,
                normalized_texts=normalized_texts,
                pinyin_texts=pinyin_texts,
                command_ids=command_ids,
                command_codes=command_codes,
                bm25=None,
            )
        tokenized = [_tokenize(text) for text in texts]
        model = BM25Okapi(tokenized) if any(tokens for tokens in tokenized) else None
        return Bm25MatcherState(
            texts=texts,
            normalized_texts=normalized_texts,
            pinyin_texts=pinyin_texts,
            command_ids=command_ids,
            command_codes=command_codes,
            bm25=model,
        )

    def _load_from_db(self, user_id: int) -> Bm25MatcherState:
        with self._session_factory() as db:
            rows: List[Command] = (
                db.query(Command)
                .filter(Command.user_id == GLOBAL_USER_ID)
                .filter(Command.status == COMMAND_STATUS_ENABLED)
                .order_by(Command.created_at.asc(), Command.id.asc())
                .all()
            )
            return self._load_bm25_state(rows)

    def get_state(self, user_id: int) -> Bm25MatcherState:
        with self._lock:
            # ignore user_id; keep a single global state
            if self._cache is not None:
                return self._cache
            state = self._load_from_db(GLOBAL_USER_ID)
            self._cache = state
            return state


class CommandService:
    # Pre-compiled regex for template matching based on zhiling.txt
    _TASK_PATTERN = re.compile(
        r"(站综合信息检查|起飞信号检查|第一次综合检查|第二次综合检查|模拟信息检查|模飞检查|对塔无线检查|下面进入基地程序|自跟踪检查|自跟踪复查|分机参数下发)"
    )
    _STAGE_PATTERN = re.compile(r"(五分钟准备|一分钟准备|停)")
    _TIMED_STAGE_PATTERN = re.compile(r"^([零一二三四五六七八九十两0-9]+)分钟准备$")
    _TIMED_STAGE_PHRASE_PATTERN = re.compile(r"[零一二三四五六七八九十两0-9]*分钟准备")
    _PREFIX_PATTERN = re.compile(r"^(各号注意[，]?|各号)")
    _VALID_STAGE_NUMBERS = frozenset({1, 5})
    _STAGE_TEXT_BY_NUMBER = {1: "一分钟准备", 5: "五分钟准备"}

    def __init__(
        self,
        *,
        session_factory=SessionLocal,
    ):
        self._session_factory = session_factory
        self._matcher = CommandMatcher(session_factory=session_factory)
        self.enable_intent_classification = os.getenv("COMMAND_INTENT_CLASSIFICATION", "true").strip().lower() in {"1", "true", "yes", "on"}

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
        # Allow multiple phrases sharing the same code. Keep as no-op for compatibility.
        return

    def get_settings(self, user_id: int) -> CommandSettings:
        with self._get_session() as db:
            settings = (
                db.query(CommandSettings)
                .filter(CommandSettings.user_id == GLOBAL_USER_ID)
                .first()
            )
            if settings:
                return settings
            settings = CommandSettings(
                user_id=GLOBAL_USER_ID,
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
                .filter(Command.user_id == GLOBAL_USER_ID)
                # Order by non-null code first, then code asc, then text, then id
                .order_by(
                    case((Command.code.is_(None), 1), else_=0).asc(),
                    Command.code.asc(),
                    Command.text.asc(),
                    Command.id.asc(),
                )
            )
            total = base_query.count()
            commands: List[Command] = (
                base_query.offset((page - 1) * page_size).limit(page_size).all()
            )
        settings = self.get_settings(GLOBAL_USER_ID)

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
                    .filter(Command.user_id == GLOBAL_USER_ID, Command.text == payload.text)
                    .one_or_none()
                )
                # Allow duplicate codes across phrases; skip uniqueness enforcement
                if existing:
                    existing.embedding = b""
                    if payload.code is not None:
                        existing.code = payload.code
                else:
                    db.add(
                        Command(
                            user_id=GLOBAL_USER_ID,
                            text=payload.text,
                            code=payload.code,
                            status=COMMAND_STATUS_ENABLED,
                            embedding=b"",
                        )
                    )
            db.commit()
        self._matcher.invalidate(GLOBAL_USER_ID)
        return len(normalized)

    def search_commands(
        self,
        user_id: int,
        keyword: Optional[str],
        *,
        code: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, object]:
        query = (keyword or "").strip()
        code_query = (code or "").strip()
        if not query and not code_query:
            return {"items": [], "total": 0, "page": 1, "page_size": page_size}
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 200))
        with self._get_session() as db:
            base_query = db.query(Command).filter(Command.user_id == GLOBAL_USER_ID)
            if query:
                like_pattern = f"%{query}%"
                base_query = base_query.filter(Command.text.like(like_pattern))
            if code_query:
                code_pattern = f"%{code_query}%"
                base_query = base_query.filter(Command.code.like(code_pattern))
            base_query = base_query.order_by(
                # Order by non-null code first, then code asc, then text, then id
                case((Command.code.is_(None), 1), else_=0).asc(),
                Command.code.asc(),
                Command.text.asc(),
                Command.id.asc(),
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
                .filter(CommandSettings.user_id == GLOBAL_USER_ID)
                .first()
            )
            if settings is None:
                settings = CommandSettings(user_id=GLOBAL_USER_ID)
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
                .filter(Command.user_id == GLOBAL_USER_ID, Command.id == command_id)
                .one_or_none()
            )
            if not command:
                return False
            db.delete(command)
            db.commit()
        self._matcher.invalidate(GLOBAL_USER_ID)
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
                .filter(Command.user_id == GLOBAL_USER_ID, Command.id == command_id)
                .one_or_none()
            )
            if not command:
                raise ValueError("Command not found")
            duplicate = (
                db.query(Command)
                .filter(Command.user_id == GLOBAL_USER_ID, Command.text == new_text, Command.id != command_id)
                .first()
            )
            if duplicate:
                raise ValueError("Command text already exists")
            if update_code:
                # Allow duplicate codes across phrases; skip uniqueness enforcement
                command.code = normalized_code
            command.text = new_text
            command.embedding = b""
            db.commit()
            db.refresh(command)
        self._matcher.invalidate(GLOBAL_USER_ID)

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
                .filter(Command.user_id == GLOBAL_USER_ID, Command.id == command_id)
                .one_or_none()
            )
            if not command:
                raise ValueError("Command not found")
            command.status = new_status
            db.commit()
            db.refresh(command)
        self._matcher.invalidate(GLOBAL_USER_ID)

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

    def _find_command_by_text(self, text: str, state: Bm25MatcherState) -> Optional[CommandMatchResult]:
        """
        Helper to find a command by exact or normalized text match in the current state.
        """
        for i, cmd_text in enumerate(state.texts):
            if text == cmd_text or text == state.normalized_texts[i]:
                return CommandMatchResult(
                    True, cmd_text, 1.0,
                    command_id=state.command_ids[i],
                    command_code=state.command_codes[i]
                )
        return None

    def _parse_stage(self, remainder: str) -> StageParseResult:
        content = (remainder or "").strip("，, ")
        if not content:
            return StageParseResult()
        stage_match = self._STAGE_PATTERN.match(content)
        if stage_match:
            return StageParseResult(stage=stage_match.group(1), has_timed_stage_phrase="分钟准备" in content)
        if "分钟准备" not in content:
            return StageParseResult()
        timed_phrase = self._TIMED_STAGE_PHRASE_PATTERN.search(content)
        if not timed_phrase:
            return StageParseResult(has_timed_stage_phrase=True, invalid_timed_stage=True)
        timed_stage = timed_phrase.group(0)
        stage_number_match = self._TIMED_STAGE_PATTERN.match(timed_stage)
        if not stage_number_match:
            return StageParseResult(has_timed_stage_phrase=True, invalid_timed_stage=True)
        stage_number = _parse_numeric_token(stage_number_match.group(1))
        if stage_number not in self._VALID_STAGE_NUMBERS:
            return StageParseResult(has_timed_stage_phrase=True, invalid_timed_stage=True)
        return StageParseResult(
            stage=self._STAGE_TEXT_BY_NUMBER[stage_number],
            has_timed_stage_phrase=True,
        )

    def match_command(
        self,
        user_id: int,
        text: str,
        *,
        threshold_override: Optional[float] = None,
        settings: Optional[CommandSettings] = None,
    ) -> CommandMatchResult:
        original_text = (text or "").strip()
        if not original_text:
            return CommandMatchResult(False, None, 0.0, original_text="", normalized_text="")
        content = normalize_command_text(original_text)
        if not content:
            return CommandMatchResult(False, None, 0.0, original_text=original_text, normalized_text="")

        intent_detected = IntentClassifier.is_intent(content) if self.enable_intent_classification else True

        # Ignore user scope; use global settings
        current_settings = settings or self.get_settings(GLOBAL_USER_ID)
        if not current_settings.enable_matching:
            return CommandMatchResult(
                False, None, 0.0,
                original_text=original_text,
                normalized_text=content,
                intent_detected=intent_detected
            )

        state = self._matcher.get_state(GLOBAL_USER_ID)
        if not isinstance(state, Bm25MatcherState):
            raise ValueError("Unexpected matcher state for BM25 backend")

        if self._has_invalid_timed_stage(content):
            logger.info("command.match result=not_matched reason=invalid_stage_number query=%s", content)
            return CommandMatchResult(
                False,
                None,
                0.0,
                original_text=original_text,
                normalized_text=content,
                intent_detected=intent_detected,
            )

        # Stage 1: Exact Match (already normalized)
        exact_match = self._find_command_by_text(content, state)
        if exact_match:
            exact_match.original_text = original_text
            exact_match.normalized_text = content
            exact_match.intent_detected = True
            exact_match.match_type = "exact"
            logger.info("command.match result=matched type=exact query=%s candidate=%s", content, exact_match.command)
            return exact_match

        # Stage 2: Rule-based / Template Match
        if self.enable_intent_classification:
            rule_match = self._match_rules(content, state)
            if rule_match:
                rule_match.original_text = original_text
                rule_match.normalized_text = content
                rule_match.intent_detected = True
                return rule_match

        # Stage 3: Fuzzy Match Fallback
        threshold = threshold_override or current_settings.match_threshold or self.default_threshold
        if self.enable_intent_classification and not intent_detected:
            # Boost threshold for non-intent text to avoid mis-matching chat to commands
            threshold = max(threshold, 0.85)

        result = self._match_with_bm25(content, state, threshold, original_text=original_text)
        result.intent_detected = intent_detected
        if result.matched:
            result.match_type = "fuzzy"
        return result

    def _has_invalid_timed_stage(self, content: str) -> bool:
        stripped = self._PREFIX_PATTERN.sub("", content).strip("，, ")
        task_match = self._TASK_PATTERN.match(stripped)
        remainder = stripped[task_match.end():].strip("，, ") if task_match else stripped
        return self._parse_stage(remainder).invalid_timed_stage

    def _match_rules(self, content: str, state: Bm25MatcherState) -> Optional[CommandMatchResult]:
        """
        Attempt to match using extracted Task and Stage patterns.
        """
        # 1. Clean prefix
        stripped = self._PREFIX_PATTERN.sub("", content).strip("，, ")
        
        # 2. Try to find Task and Stage using match() to ensure they start at the beginning of the stripped text
        task_match = self._TASK_PATTERN.match(stripped)
        
        if task_match:
            task = task_match.group(1)
            # Find stage in the remainder of the string
            remainder = stripped[task_match.end():].strip("，, ")
            stage_result = self._parse_stage(remainder)
            if stage_result.invalid_timed_stage:
                logger.info("command.match result=not_matched reason=invalid_stage_number query=%s", content)
                return None
            stage = stage_result.stage
            
            # Reconstruction attempts
            candidates = [f"各号注意，{task}{stage}", f"{task}{stage}"] if stage else []
            if not stage_result.has_timed_stage_phrase and not remainder:
                candidates.extend([f"各号注意，{task}", f"{task}"])
            
            for cand in candidates:
                res = self._find_command_by_text(cand, state)
                if res:
                    res.score = 0.95
                    res.match_type = "rule"
                    logger.info("command.match result=matched type=rule query=%s candidate=%s", content, res.command)
                    return res
        
        # 3. Special case for independent stages or single words
        if content in {"起飞", "点火", "发射", "停", "五分钟准备", "一分钟准备"}:
             res = self._find_command_by_text(content, state)
             if res:
                 res.match_type = "exact"
                 return res

        return None

    def _match_with_bm25(
        self,
        content: str,
        state: Bm25MatcherState,
        threshold: float,
        *,
        original_text: Optional[str] = None,
    ) -> CommandMatchResult:
        raw_text = (original_text or content or "").strip()
        if not state.texts or state.bm25 is None:
            return CommandMatchResult(False, None, 0.0, original_text=raw_text, normalized_text=content)
        normalized_query = _normalize_for_matching(content)
        query_pinyin = _build_pinyin(content) if PHONETIC_ENABLED else ""
        query_tokens = _tokenize(content)
        query_ordinals, query_numbers = _extract_numeric_tokens(content)
        if not query_tokens and not query_pinyin:
            return CommandMatchResult(False, None, 0.0, original_text=raw_text, normalized_text=content)
        candidate_indices = set()
        if state.bm25 is not None and query_tokens:
            scores = np.asarray(state.bm25.get_scores(query_tokens), dtype=np.float32)
            if scores.size:
                top_k = min(BM25_TOP_K, scores.size)
                bm25_indices = np.argsort(scores)[-top_k:][::-1]
                candidate_indices.update(int(idx) for idx in bm25_indices)
        pinyin_scores: Optional[np.ndarray] = None
        if PHONETIC_ENABLED and query_pinyin:
            if state.pinyin_texts:
                pinyin_values = [
                    float(fuzz.token_set_ratio(query_pinyin, candidate_pinyin))
                    * _pinyin_length_ratio(query_pinyin, candidate_pinyin)
                    if candidate_pinyin
                    else 0.0
                    for candidate_pinyin in state.pinyin_texts
                ]
                pinyin_scores = np.asarray(pinyin_values, dtype=np.float32)
                if pinyin_scores.size:
                    max_pinyin_score = float(np.max(pinyin_scores))
                    if max_pinyin_score > 0.0:
                        top_k = min(BM25_TOP_K, pinyin_scores.size)
                        phonetic_indices = np.argsort(pinyin_scores)[-top_k:][::-1]
                        candidate_indices.update(int(idx) for idx in phonetic_indices)
        if not candidate_indices:
            logger.info("command.match no_candidate query=%s", content)
            return CommandMatchResult(False, None, 0.0, original_text=raw_text, normalized_text=content)
        best_score = 0.0
        best_text_score = 0.0
        best_pinyin_score: Optional[float] = None
        best_final_score = 0.0
        best_text: Optional[str] = None
        best_code: Optional[str] = None
        best_id: Optional[int] = None
        best_candidate_pinyin: Optional[str] = None
        best_numeric_factor = 1.0
        best_query_ordinals: Tuple[int, ...] = query_ordinals
        best_query_numbers: Tuple[int, ...] = query_numbers
        best_candidate_ordinals: Tuple[int, ...] = ()
        best_candidate_numbers: Tuple[int, ...] = ()
        for idx_int in candidate_indices:
            candidate_text = state.texts[idx_int]
            candidate_normalized = state.normalized_texts[idx_int]
            text_score = float(fuzz.token_set_ratio(normalized_query, candidate_normalized))
            final_score = text_score
            pinyin_score: Optional[float] = None
            candidate_pinyin: Optional[str] = None
            if PHONETIC_ENABLED and query_pinyin:
                candidate_pinyin = state.pinyin_texts[idx_int] if idx_int < len(state.pinyin_texts) else ""
                if candidate_pinyin:
                    if pinyin_scores is not None:
                        pinyin_score = float(pinyin_scores[idx_int])
                    else:
                        pinyin_score = float(fuzz.token_set_ratio(query_pinyin, candidate_pinyin)) * _pinyin_length_ratio(
                            query_pinyin, candidate_pinyin
                        )
                    blended_score = (1.0 - PHONETIC_WEIGHT) * text_score + PHONETIC_WEIGHT * pinyin_score
                    final_score = max(text_score, pinyin_score, blended_score)
            candidate_ordinals, candidate_numbers = _extract_numeric_tokens(candidate_text)
            numeric_factor = _numeric_factor(
                query_ordinals,
                query_numbers,
                candidate_ordinals,
                candidate_numbers,
            )
            final_score *= numeric_factor
            if final_score > best_score:
                best_score = final_score
                best_text_score = text_score
                best_pinyin_score = pinyin_score
                best_final_score = final_score
                best_text = candidate_text
                best_code = state.command_codes[idx_int] if idx_int < len(state.command_codes) else None
                best_id = state.command_ids[idx_int] if idx_int < len(state.command_ids) else None
                best_candidate_pinyin = candidate_pinyin if pinyin_score is not None else None
                best_numeric_factor = numeric_factor
                best_candidate_ordinals = candidate_ordinals
                best_candidate_numbers = candidate_numbers
        normalized = best_final_score / 100.0
        if not best_text:
            logger.info("command.match no_candidate query=%s", content)
            return CommandMatchResult(False, None, normalized, original_text=raw_text, normalized_text=content)
        text_norm = best_text_score / 100.0
        pinyin_norm = best_pinyin_score / 100.0 if best_pinyin_score is not None else None
        if text_norm >= threshold:
            logger.info(
                "command.match result=matched type=text query=%s candidate=%s id=%s code=%s text_score=%.2f final_score=%.2f pinyin_score=%s numeric_factor=%.2f threshold=%.2f query_pinyin=%s candidate_pinyin=%s query_ordinals=%s candidate_ordinals=%s query_numbers=%s candidate_numbers=%s",
                content,
                best_text,
                best_id,
                best_code,
                best_text_score,
                best_final_score,
                f"{best_pinyin_score:.2f}" if best_pinyin_score is not None else "n/a",
                best_numeric_factor,
                threshold,
                query_pinyin or "",
                best_candidate_pinyin or "",
                ",".join(str(value) for value in best_query_ordinals),
                ",".join(str(value) for value in best_candidate_ordinals),
                ",".join(str(value) for value in best_query_numbers),
                ",".join(str(value) for value in best_candidate_numbers),
            )
            return CommandMatchResult(
                True,
                best_text,
                normalized,
                command_id=best_id,
                command_code=best_code,
                original_text=raw_text,
                normalized_text=content,
            )
        if PHONETIC_ENABLED and pinyin_norm is not None and pinyin_norm >= PHONETIC_THRESHOLD:
            logger.info(
                "command.match result=matched type=phonetic query=%s candidate=%s id=%s code=%s text_score=%.2f final_score=%.2f pinyin_score=%s numeric_factor=%.2f threshold=%.2f query_pinyin=%s candidate_pinyin=%s query_ordinals=%s candidate_ordinals=%s query_numbers=%s candidate_numbers=%s",
                content,
                best_text,
                best_id,
                best_code,
                best_text_score,
                best_final_score,
                f"{best_pinyin_score:.2f}" if best_pinyin_score is not None else "n/a",
                best_numeric_factor,
                threshold,
                query_pinyin or "",
                best_candidate_pinyin or "",
                ",".join(str(value) for value in best_query_ordinals),
                ",".join(str(value) for value in best_candidate_ordinals),
                ",".join(str(value) for value in best_query_numbers),
                ",".join(str(value) for value in best_candidate_numbers),
            )
            return CommandMatchResult(
                True,
                best_text,
                normalized,
                command_id=best_id,
                command_code=best_code,
                original_text=raw_text,
                normalized_text=content,
            )
        if normalized < threshold:
            logger.info(
                "command.match result=not_matched type=below_threshold query=%s candidate=%s id=%s code=%s text_score=%.2f final_score=%.2f pinyin_score=%s numeric_factor=%.2f threshold=%.2f query_pinyin=%s candidate_pinyin=%s query_ordinals=%s candidate_ordinals=%s query_numbers=%s candidate_numbers=%s",
                content,
                best_text,
                best_id,
                best_code,
                best_text_score,
                best_final_score,
                f"{best_pinyin_score:.2f}" if best_pinyin_score is not None else "n/a",
                best_numeric_factor,
                threshold,
                query_pinyin or "",
                best_candidate_pinyin or "",
                ",".join(str(value) for value in best_query_ordinals),
                ",".join(str(value) for value in best_candidate_ordinals),
                ",".join(str(value) for value in best_query_numbers),
                ",".join(str(value) for value in best_candidate_numbers),
            )
            return CommandMatchResult(False, None, normalized, original_text=raw_text, normalized_text=content)
        logger.info(
            "command.match result=matched type=blend query=%s candidate=%s id=%s code=%s text_score=%.2f final_score=%.2f pinyin_score=%s numeric_factor=%.2f threshold=%.2f query_pinyin=%s candidate_pinyin=%s query_ordinals=%s candidate_ordinals=%s query_numbers=%s candidate_numbers=%s",
            content,
            best_text,
            best_id,
            best_code,
            best_text_score,
            best_final_score,
            f"{best_pinyin_score:.2f}" if best_pinyin_score is not None else "n/a",
            best_numeric_factor,
            threshold,
            query_pinyin or "",
            best_candidate_pinyin or "",
            ",".join(str(value) for value in best_query_ordinals),
            ",".join(str(value) for value in best_candidate_ordinals),
            ",".join(str(value) for value in best_query_numbers),
            ",".join(str(value) for value in best_candidate_numbers),
        )
        return CommandMatchResult(
            True,
            best_text,
            normalized,
            command_id=best_id,
            command_code=best_code,
            original_text=raw_text,
            normalized_text=content,
        )


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
