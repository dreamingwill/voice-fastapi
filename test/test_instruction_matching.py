import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Path to the local BAAI/bge-small-zh SentenceTransformer checkpoint.
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "bge-small-zh"

# python test/test_instruction_matching.py
# Example instruction inventory used to seed the SQLite database for testing.
INSTRUCTIONS = [
    "各号注意，站综合信息检查五分钟准备",
    "站综合信息检查一分钟准备",
    "开始站务系统登录，确认终端在线",
    "应急演练：进入静默模式等待命令",
]

# Cosine similarity threshold under which we treat the result as "no match".
MATCH_THRESHOLD = 0.9


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instructions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()


def encode_texts(model: SentenceTransformer, texts) -> np.ndarray:
    """Return L2-normalized float32 embeddings for the provided texts."""
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def insert_instructions(
    conn: sqlite3.Connection,
    model: SentenceTransformer,
    instructions,
) -> None:
    ensure_schema(conn)
    vectors = encode_texts(model, instructions)
    conn.executemany(
        "INSERT OR REPLACE INTO instructions(text, embedding) VALUES (?, ?)",
        [
            (text, vector.tobytes())
            for text, vector in zip(instructions, vectors)
        ],
    )
    conn.commit()


def fetch_instruction_vectors(
    conn: sqlite3.Connection,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    rows = conn.execute("SELECT text, embedding FROM instructions").fetchall()
    if not rows:
        return np.empty((0, 512), dtype=np.float32), tuple()
    texts = tuple(row[0] for row in rows)
    vectors = np.vstack(
        [np.frombuffer(row[1], dtype=np.float32) for row in rows]
    )
    return vectors, texts


def find_best_match(
    model: SentenceTransformer,
    conn: sqlite3.Connection,
    query: str,
    threshold: float = MATCH_THRESHOLD,
) -> Tuple[Optional[str], float]:
    query_vec = encode_texts(model, [query])[0]
    vectors, texts = fetch_instruction_vectors(conn)
    if len(texts) == 0:
        return None, 0.0

    scores = vectors @ query_vec  # cosine similarity because vectors are normalized
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < threshold:
        return None, best_score
    return texts[best_idx], best_score


def run_demo() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"model path not found: {MODEL_DIR}")

    model = SentenceTransformer(str(MODEL_DIR))
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        insert_instructions(conn, model, INSTRUCTIONS)

        expected: Dict[str, Optional[str]] = {
            "站综合信息检查一分钟备": "站综合信息检查一分钟准备",
            "各号注意，站综合信息检查五分钟准备": "各号注意，站综合信息检查五分钟准备",
            "现在进行站务系统登录检查": "开始站务系统登录，确认终端在线",
            "静默等待进一步命令": "应急演练：进入静默模式等待命令",
            "今天的天气真好，大家休息": None,
        }

        for query, target in expected.items():
            match, score = find_best_match(model, conn, query)
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
