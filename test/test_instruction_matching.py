import sqlite3
import tempfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

try:
    import jieba
except ImportError:  # pragma: no cover - dependency installed in production
    jieba = None

# Example instruction inventory used to seed the SQLite database for testing.
INSTRUCTIONS = [
    "各号注意，站综合信息检查五分钟准备",
    "站综合信息检查一分钟准备",
    "开始站务系统登录，确认终端在线",
    "应急演练：进入静默模式等待命令",
]

# RapidFuzz 返回 0–100，除以 100 后与阈值比较。
MATCH_THRESHOLD = 0.75
BM25_TOP_K = 10


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instructions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL UNIQUE
        )
        """
    )
    conn.commit()


def _normalize(text: str) -> str:
    return (text or "").strip().casefold()


def _tokenize(text: str) -> Sequence[str]:
    normalized = _normalize(text)
    if not normalized:
        return []
    if jieba is not None:
        return [token.strip() for token in jieba.cut(normalized, cut_all=False) if token.strip()]
    return [char for char in normalized if not char.isspace()]


def insert_instructions(conn: sqlite3.Connection, instructions: Sequence[str]) -> None:
    ensure_schema(conn)
    conn.executemany(
        "INSERT OR REPLACE INTO instructions(text) VALUES (?)",
        [(text,) for text in instructions],
    )
    conn.commit()


def fetch_instructions(conn: sqlite3.Connection) -> Tuple[str, ...]:
    rows = conn.execute("SELECT text FROM instructions ORDER BY id ASC").fetchall()
    return tuple(row[0] for row in rows)


def build_bm25_state(texts: Tuple[str, ...]) -> Tuple[Tuple[str, ...], Tuple[str, ...], Optional[BM25Okapi]]:
    normalized = tuple(_normalize(text) for text in texts)
    tokenized = [_tokenize(text) for text in texts]
    if not texts or not any(tokenized):
        return texts, normalized, None
    return texts, normalized, BM25Okapi(tokenized)


def find_best_match(
    texts: Tuple[str, ...],
    normalized_texts: Tuple[str, ...],
    bm25: Optional[BM25Okapi],
    query: str,
    threshold: float = MATCH_THRESHOLD,
) -> Tuple[Optional[str], float]:
    if not texts or bm25 is None:
        return None, 0.0
    normalized_query = _normalize(query)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return None, 0.0

    scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
    if scores.size == 0:
        return None, 0.0

    top_k = min(BM25_TOP_K, scores.size)
    sorted_indices = np.argsort(scores)[-top_k:][::-1]
    best_score = 0.0
    best_text: Optional[str] = None

    for idx in sorted_indices:
        idx_int = int(idx)
        candidate_text = texts[idx_int]
        candidate_normalized = normalized_texts[idx_int]
        fuzzy_score = float(fuzz.token_set_ratio(normalized_query, candidate_normalized))
        if fuzzy_score > best_score:
            best_score = fuzzy_score
            best_text = candidate_text

    normalized_score = best_score / 100.0
    if not best_text or normalized_score < threshold:
        return None, normalized_score
    return best_text, normalized_score


def run_demo() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        insert_instructions(conn, INSTRUCTIONS)

        texts = fetch_instructions(conn)
        texts, normalized, bm25 = build_bm25_state(texts)

        expected: Dict[str, Optional[str]] = {
            "站综合信息检查一分钟备": "站综合信息检查一分钟准备",
            "各号注意，站综合信息检查五分钟准备": "各号注意，站综合信息检查五分钟准备",
            "现在进行站务系统登录检查": "开始站务系统登录，确认终端在线",
            "静默等待进一步命令": "应急演练：进入静默模式等待命令",
            "今天的天气真好，大家休息": None,
        }

        for query, target in expected.items():
            match, score = find_best_match(texts, normalized, bm25, query)
            if target is None:
                if match is None:
                    print(f"[不匹配] '{query}' -> 不匹配 (score={score:.3f})")
                else:
                    print(
                        "[警告] 期望不匹配但获得匹配："
                        f"'{query}' -> '{match}' (score={score:.3f})"
                    )
            else:
                if match == target:
                    print(f"[匹配] '{query}' -> '{match}' (score={score:.3f})")
                else:
                    print(
                        "[警告] 匹配结果与期望不符："
                        f"'{query}' -> '{match}' (score={score:.3f})"
                    )


if __name__ == "__main__":
    run_demo()
