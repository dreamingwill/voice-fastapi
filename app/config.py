import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./database/voiceprints.db")
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "./recordings/")
RECORDINGS_MAX_COUNT = int(os.getenv("RECORDINGS_MAX_COUNT", "10"))

ACCESS_TOKEN_TTL = int(os.getenv("ACCESS_TOKEN_TTL", "3600"))
REFRESH_TOKEN_TTL = int(os.getenv("REFRESH_TOKEN_TTL", str(7 * 24 * 3600)))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "voice123")
ADMIN_DISPLAY_NAME = os.getenv("ADMIN_DISPLAY_NAME", "系统管理员")
ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

DEFAULT_CONFIG_PATH = os.getenv("VOICE_SERVER_CONFIG", "config/app_config.json")
COMMAND_FORWARD_URL = os.getenv("COMMAND_FORWARD_URL", "")
COMMAND_FORWARD_TIMEOUT = float(os.getenv("COMMAND_FORWARD_TIMEOUT", "5"))
print("当前 COMMAND_FORWARD_URL =", COMMAND_FORWARD_URL)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

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
    parser.add_argument(
        "--asr_mode",
        type=str,
        default="streaming_transducer",
        choices=["streaming_transducer", "offline_sense_voice"],
    )
    parser.add_argument("--tokens", type=str)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--decoder", type=str)
    parser.add_argument("--joiner", type=str)
    parser.add_argument("--sense_voice_model", type=str, default="")
    parser.add_argument("--sense_voice_use_itn", type=_parse_bool, default=True)
    parser.add_argument("--sense_voice_language", type=str, default="auto")
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
    parser.add_argument("--rule1_min_trailing_silence", type=float, default=0.8)
    parser.add_argument("--rule2_min_trailing_silence", type=float, default=0.4)
    parser.add_argument("--rule3_min_utterance_length", type=int, default=15)
    parser.add_argument("--command_forward_url", type=str, default="")
    parser.add_argument("--command_forward_timeout", type=float, default=5.0)
    parser.add_argument(
        "--command_intent_classification",
        type=_parse_bool,
        default=os.getenv("COMMAND_INTENT_CLASSIFICATION", "true").strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument("--vad_pre_roll_ms", type=int, default=int(os.getenv("PRE_ROLL_MS", "300")))
    parser.add_argument("--vad_post_roll_ms", type=int, default=int(os.getenv("POST_ROLL_MS", "700")))
    parser.add_argument("--vad_snr_open_db", type=float, default=float(os.getenv("SNR_OPEN_DB", "10")))
    parser.add_argument("--vad_open_min_ms", type=int, default=int(os.getenv("OPEN_MIN_MS", "120")))
    parser.add_argument("--vad_end_silence_ms", type=int, default=int(os.getenv("END_SILENCE_MS", "900")))
    parser.add_argument("--vad_max_utterance_ms", type=int, default=int(os.getenv("MAX_UTTERANCE_MS", "0")))
    parser.add_argument("--vad_reopen_min_ms", type=int, default=int(os.getenv("VAD_REOPEN_MIN_MS", "120")))
    parser.add_argument("--vad_noise_margin_db", type=float, default=float(os.getenv("VAD_NOISE_MARGIN_DB", "3")))
    parser.add_argument(
        "--vad_noise_bootstrap_ms",
        type=int,
        default=int(os.getenv("VAD_NOISE_BOOTSTRAP_MS", "1000")),
    )
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
    global COMMAND_FORWARD_URL, COMMAND_FORWARD_TIMEOUT

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

    if "COMMAND_FORWARD_URL" in os.environ:
        args.command_forward_url = os.environ["COMMAND_FORWARD_URL"]
    if "COMMAND_FORWARD_TIMEOUT" in os.environ:
        args.command_forward_timeout = float(os.environ["COMMAND_FORWARD_TIMEOUT"])

    COMMAND_FORWARD_URL = args.command_forward_url.strip()
    COMMAND_FORWARD_TIMEOUT = float(args.command_forward_timeout)
    return args
