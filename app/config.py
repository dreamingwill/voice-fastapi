import argparse
import os


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./database/voiceprints.db")

ACCESS_TOKEN_TTL = int(os.getenv("ACCESS_TOKEN_TTL", "3600"))
REFRESH_TOKEN_TTL = int(os.getenv("REFRESH_TOKEN_TTL", str(7 * 24 * 3600)))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "voice123")
ADMIN_DISPLAY_NAME = os.getenv("ADMIN_DISPLAY_NAME", "指挥员 张伟")
ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def parse_args():
    parser = argparse.ArgumentParser(description="Speech Recognition Server")

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sample_rate", type=int)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--tokens", type=str)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--decoder", type=str)
    parser.add_argument("--joiner", type=str)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=80)
    parser.add_argument("--decoding_method", type=str, default="greedy_search")
    parser.add_argument("--max_active_paths", type=int, default=4)
    parser.add_argument("--provider", type=str, default="cpu")
    parser.add_argument("--hotwords_file", type=str, default="")
    parser.add_argument("--hotwords_score", type=float, default=1.5)
    parser.add_argument("--blank_penalty", type=float, default=0.0)
    parser.add_argument("--hr_lexicon", type=str, default="")
    parser.add_argument("--hr_rule_fsts", type=str, default="")
    parser.add_argument("--min_spk_seconds", type=float, default=1.5)

    return parser.parse_args()
