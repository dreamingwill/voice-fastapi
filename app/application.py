import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import auth as auth_routes
from .api import enhancement as enhancement_routes
from .api import job_positions as job_positions_routes
from .api import logs as logs_routes
from .api import settings as settings_routes
from .api import status as status_routes
from .api import users as users_routes
from .api import ws as ws_routes
from .api import transcripts as transcripts_routes
from .api import commands as command_routes
from .config import DEFAULT_ALLOWED_ORIGINS
from .database import SessionLocal, init_db
from .services.settings import load_system_settings_snapshot
from .services.audio_enhancement import AudioEnhancementPipeline
from .services.voice import SpeakerEmbedder
from .services.voice.asr_engines import get_asr_mode, get_asr_model_path
from .utils import now_utc


def validate_asr_args(args) -> None:
    asr_mode = get_asr_mode(args)
    tokens = getattr(args, "tokens", "") or ""

    if not tokens:
        raise ValueError("ASR tokens path is required")

    if asr_mode == "streaming_transducer":
        missing = [
            name
            for name in ("encoder", "decoder", "joiner")
            if not (getattr(args, name, "") or "")
        ]
        if missing:
            raise ValueError(f"streaming_transducer requires: {', '.join(missing)}")
        return

    if asr_mode != "offline_sense_voice":
        raise ValueError(f"Unsupported ASR mode: {asr_mode}")

    sense_voice_model = getattr(args, "sense_voice_model", "") or ""
    if not sense_voice_model:
        raise ValueError("offline_sense_voice requires sense_voice_model")

    vad_max_utterance_ms = getattr(args, "vad_max_utterance_ms", 0)
    if not isinstance(vad_max_utterance_ms, (int, float)) or vad_max_utterance_ms <= 0:
        raise ValueError("offline_sense_voice requires vad_max_utterance_ms > 0")

    provider = (getattr(args, "provider", "cpu") or "cpu").lower()
    if provider == "rknn" and not sense_voice_model.endswith(".rknn"):
        raise ValueError("offline_sense_voice with provider=rknn requires a .rknn model")
    if provider != "rknn" and not sense_voice_model.endswith(".onnx"):
        raise ValueError("offline_sense_voice with non-rknn provider requires a .onnx model")


def _log_asr_startup(logger: logging.Logger, args) -> None:
    asr_mode = get_asr_mode(args)
    asr_model = get_asr_model_path(args)

    if asr_mode == "offline_sense_voice":
        logger.info(
            (
                "initializing ASR mode=%s host=%s port=%s model=%s tokens=%s provider=%s "
                "sample_rate=%s feature_dim=%s decoding=%s threads=%s use_itn=%s language=%s "
                "hr_rule_fsts=%s hr_lexicon=%s vad_pre=%s vad_post=%s vad_snr=%.1f "
                "vad_open=%s vad_end=%s vad_max=%s vad_reopen=%s vad_noise_margin=%.1f "
                "vad_noise_bootstrap=%s"
            ),
            asr_mode,
            args.host,
            args.port,
            asr_model,
            args.tokens,
            args.provider,
            args.sample_rate,
            args.feature_dim,
            args.decoding_method,
            args.num_threads,
            args.sense_voice_use_itn,
            args.sense_voice_language,
            args.hr_rule_fsts or "",
            args.hr_lexicon or "",
            args.vad_pre_roll_ms,
            args.vad_post_roll_ms,
            args.vad_snr_open_db,
            args.vad_open_min_ms,
            args.vad_end_silence_ms,
            args.vad_max_utterance_ms,
            args.vad_reopen_min_ms,
            args.vad_noise_margin_db,
            args.vad_noise_bootstrap_ms,
        )
        return

    logger.info(
        (
            "initializing ASR mode=%s host=%s port=%s tokens=%s encoder=%s "
            "decoder=%s joiner=%s provider=%s sample_rate=%s feature_dim=%s "
            "decoding=%s max_paths=%s threads=%s hotwords_file=%s "
            "hotwords_score=%.2f blank_penalty=%.2f hr_rule_fsts=%s hr_lexicon=%s "
            "rule1=%.3f rule2=%.3f rule3=%s vad_pre=%s vad_post=%s vad_snr=%.1f "
            "vad_open=%s vad_end=%s vad_max=%s vad_reopen=%s vad_noise_margin=%.1f "
            "vad_noise_bootstrap=%s"
        ),
        asr_mode,
        args.host,
        args.port,
        args.tokens,
        args.encoder,
        args.decoder,
        args.joiner,
        args.provider,
        args.sample_rate,
        args.feature_dim,
        args.decoding_method,
        args.max_active_paths,
        args.num_threads,
        args.hotwords_file or "",
        args.hotwords_score,
        args.blank_penalty,
        args.hr_rule_fsts or "",
        args.hr_lexicon or "",
        args.rule1_min_trailing_silence,
        args.rule2_min_trailing_silence,
        args.rule3_min_utterance_length,
        args.vad_pre_roll_ms,
        args.vad_post_roll_ms,
        args.vad_snr_open_db,
        args.vad_open_min_ms,
        args.vad_end_silence_ms,
        args.vad_max_utterance_ms,
        args.vad_reopen_min_ms,
        args.vad_noise_margin_db,
        args.vad_noise_bootstrap_ms,
    )


def create_app(args):
    validate_asr_args(args)
    init_db()

    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

    logger = logging.getLogger("app.startup")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.args = args
        app.state.start_time = now_utc()
        app.state.session_metrics = {
            "active_sessions": 0,
            "latency_samples": [],
            "similarity_samples": [],
            "undetermined_count": 0,
            "audio_queue_depth": 0,
        }
        app.state.enhancement_pipeline = AudioEnhancementPipeline()
        with SessionLocal() as db:
            app.state.system_settings = load_system_settings_snapshot(db)

        _log_asr_startup(logger, args)

        app.state.asr_ready = True
        app.state.embedder = SpeakerEmbedder(
            model_path=args.model_path,
            sample_rate=args.sample_rate,
            threshold=args.threshold,
        )
        logger.info(
            "initializing embedder model=%s sample_rate=%s threshold=%.3f min_spk_seconds=%.2f",
            args.model_path,
            args.sample_rate,
            args.threshold,
            args.min_spk_seconds,
        )
        try:
            yield
        finally:
            app.state.asr_ready = False
            app.state.embedder = None
            app.state.enhancement_pipeline = None

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_routes.router)
    app.include_router(users_routes.router)
    app.include_router(logs_routes.router)
    app.include_router(status_routes.router)
    app.include_router(transcripts_routes.router)
    app.include_router(command_routes.router)
    app.include_router(settings_routes.router)
    app.include_router(job_positions_routes.router)
    app.include_router(enhancement_routes.router)
    app.include_router(ws_routes.router)

    @app.get("/")
    async def root():
        return {"status": "ok"}

    return app
