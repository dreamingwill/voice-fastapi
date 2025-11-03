import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import auth as auth_routes
from .api import logs as logs_routes
from .api import status as status_routes
from .api import users as users_routes
from .api import ws as ws_routes
from .config import DEFAULT_ALLOWED_ORIGINS
from .database import init_db
from .services.voice import SpeakerEmbedder, create_recognizer
from .utils import now_utc


def create_app(args):
    init_db()

    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

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

        app.state.recognizer = create_recognizer(
            tokens=args.tokens,
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            max_active_paths=args.max_active_paths,
            provider=args.provider,
            hotwords_file=args.hotwords_file,
            hotwords_score=args.hotwords_score,
            blank_penalty=args.blank_penalty,
            hr_rule_fsts=args.hr_rule_fsts,
            hr_lexicon=args.hr_lexicon,
        )
        app.state.embedder = SpeakerEmbedder(
            model_path=args.model_path,
            sample_rate=args.sample_rate,
            threshold=args.threshold,
        )
        try:
            yield
        finally:
            app.state.recognizer = None
            app.state.embedder = None

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
    app.include_router(ws_routes.router)

    return app
