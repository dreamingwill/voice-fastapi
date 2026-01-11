import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import auth as auth_routes
from .api import enhancement as enhancement_routes
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
from .services.voice import create_speaker_embedder
from .utils import now_utc


def create_app(args):
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

        logger.info(
            (
                "initializing recognizer host=%s port=%s tokens=%s encoder=%s "
                "decoder=%s joiner=%s provider=%s sample_rate=%s feature_dim=%s "
                "decoding=%s max_paths=%s threads=%s hotwords_file=%s "
                "hotwords_score=%.2f blank_penalty=%.2f hr_rule_fsts=%s hr_lexicon=%s "
                "rule1=%.3f rule2=%.3f rule3=%s"
            ),
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
        )
        logger.info("database_url=%s", getattr(args, "database_url", None))

        app.state.asr_ready = True
        app.state.embedder = create_speaker_embedder(
            model_path=args.model_path,
            sample_rate=args.sample_rate,
            threshold=args.threshold,
            provider=args.speaker_provider,
            num_threads=args.speaker_num_threads,
            rknn_feature_dim=args.speaker_rknn_feature_dim,
            rknn_num_frames=args.speaker_rknn_num_frames,
            rknn_frame_length_ms=args.speaker_rknn_frame_length_ms,
            rknn_frame_shift_ms=args.speaker_rknn_frame_shift_ms,
            rknn_core=args.speaker_rknn_core,
            rknn_l2_normalize=args.speaker_rknn_l2_normalize,
        )
        logger.info(
            (
            "initializing embedder model=%s sample_rate=%s threshold=%.3f "
            "min_spk_seconds=%.2f provider=%s threads=%s rknn_core=%s"
        ),
            args.model_path,
            args.sample_rate,
            args.threshold,
            args.min_spk_seconds,
            args.speaker_provider,
            args.speaker_num_threads,
            args.speaker_rknn_core,
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
    app.include_router(enhancement_routes.router)
    app.include_router(ws_routes.router)

    @app.get("/")
    async def root():
        return {"status": "ok"}

    return app
