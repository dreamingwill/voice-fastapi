import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./database/voiceprints.db")

ACCESS_TOKEN_TTL = int(os.getenv("ACCESS_TOKEN_TTL", "3600"))
REFRESH_TOKEN_TTL = int(os.getenv("REFRESH_TOKEN_TTL", str(7 * 24 * 3600)))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "voice123")
ADMIN_DISPLAY_NAME = os.getenv("ADMIN_DISPLAY_NAME", "系统管理员")
ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

DEFAULT_CONFIG_PATH = os.getenv("VOICE_SERVER_CONFIG", "config/app_config.json")


def _load_config(path: Optional[str], *, required: bool = False) -> Dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        if required:
            raise FileNotFoundError(f"Config file not found: {file_path}")
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config: {file_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {file_path}")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speech Recognition Server",
        parents=[_config_parser()],
        allow_abbrev=False,
    )
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
    parser.add_argument("--rule1_min_trailing_silence", type=float, default=2.4)
    parser.add_argument("--rule2_min_trailing_silence", type=float, default=1.2)
    parser.add_argument("--rule3_min_utterance_length", type=int, default=300)
    return parser


def _config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON config file whose values serve as defaults for CLI flags.",
    )
    return parser


def parse_args(argv: Optional[Any] = None):
    config_only = _config_parser()
    config_args, remaining = config_only.parse_known_args(argv)
    config_path = config_args.config or DEFAULT_CONFIG_PATH
    config_required = bool(config_args.config)
    config_values = _load_config(config_path, required=config_required)

    parser = _build_parser()
    if config_values:
        parser.set_defaults(**config_values)

    args = parser.parse_args(remaining)
    if config_values:
        args.config = config_path
    return args
