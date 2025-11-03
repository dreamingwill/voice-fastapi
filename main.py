from fastapi import FastAPI, WebSocket, WebSocketDisconnect, \
    HTTPException, Query, Depends, UploadFile, Response
from fastapi import status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, func, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import json
import io
import asyncio
import soundfile as sf
import random
import argparse
from datetime import datetime, timedelta, timezone
import os
import secrets

import uvicorn
import sherpa_onnx

DATABASE_URL = "sqlite:///./database/voiceprints.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    identity = Column(String, nullable=True)
    embedding = Column(Text, nullable=True)


class EventLog(Base):
    __tablename__ = "event_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    username = Column(String, nullable=True)
    operator = Column(String, nullable=True)
    type = Column(String, nullable=False)
    category = Column(String, nullable=True)
    authorized = Column(Boolean, default=True)
    payload = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


Base.metadata.create_all(bind=engine)

class UserCreateAndUpdate(BaseModel):
    username: str
    identity: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    identity: Optional[str] = None
    has_voiceprint: bool

class IdentifyResponse(BaseModel):
    matched: str
    similarity: float
    topk: list
    threshold: float

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenPayload(BaseModel):
    id: int
    username: str
    role: str
    display_name: Optional[str] = None

class TokenResponse(BaseModel):
    token: str
    expires_in: int
    user: TokenPayload
    refresh_token: str

class RefreshRequest(BaseModel):
    refresh_token: str

class RefreshResponse(BaseModel):
    token: str
    expires_in: int
    refresh_token: Optional[str] = None

class LogEntryResponse(BaseModel):
    id: int
    timestamp: str
    type: str
    session_id: Optional[str] = None
    operator: Optional[str] = None
    authorized: Optional[bool] = None
    payload: Optional[Dict[str, Any]] = None
    username: Optional[str] = None
    user_id: Optional[int] = None
    category: Optional[str] = None

class LogsResponse(BaseModel):
    items: List[LogEntryResponse]
    total: int
    page: int
    page_size: int

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: int
    asr_model: Optional[str]
    speaker_model: Optional[str]
    asr_ready: bool
    speaker_ready: bool
    database: str

class MetricsResponse(BaseModel):
    active_sessions: int
    avg_recognition_latency_ms: float
    avg_embedding_similarity: float
    undetermined_speaker_rate: float
    audio_queue_depth: int


ACCESS_TOKEN_TTL = int(os.getenv("ACCESS_TOKEN_TTL", "3600"))
REFRESH_TOKEN_TTL = int(os.getenv("REFRESH_TOKEN_TTL", str(7 * 24 * 3600)))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "voice123")
ADMIN_DISPLAY_NAME = os.getenv("ADMIN_DISPLAY_NAME", "指挥员 张伟")
ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")

security = HTTPBearer(auto_error=False)

access_tokens: Dict[str, Dict[str, Any]] = {}
refresh_tokens: Dict[str, Dict[str, Any]] = {}
token_lock = asyncio.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


async def _issue_tokens(user_payload: TokenPayload) -> TokenResponse:
    access_token = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(40)
    now = _now()
    access_exp = now + timedelta(seconds=ACCESS_TOKEN_TTL)
    refresh_exp = now + timedelta(seconds=REFRESH_TOKEN_TTL)

    record = {
        "token": access_token,
        "user": user_payload.dict(),
        "expires_at": access_exp,
    }
    refresh_record = {
        "refresh_token": refresh_token,
        "user": user_payload.dict(),
        "expires_at": refresh_exp,
    }

    async with token_lock:
        access_tokens[access_token] = record
        refresh_tokens[refresh_token] = refresh_record

    return TokenResponse(
        token=access_token,
        expires_in=ACCESS_TOKEN_TTL,
        user=user_payload,
        refresh_token=refresh_token,
    )


async def _refresh_access_token(refresh_token: str) -> TokenResponse:
    now = _now()
    async with token_lock:
        record = refresh_tokens.get(refresh_token)
        if not record:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
        if record["expires_at"] <= now:
            del refresh_tokens[refresh_token]
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Expired refresh token")

        # issue new access token
        user_payload = TokenPayload(**record["user"])

    response = await _issue_tokens(user_payload)
    return response


async def _validate_access_token(token: str) -> TokenPayload:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing access token")

    now = _now()
    async with token_lock:
        record = access_tokens.get(token)
        if not record:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token")
        if record["expires_at"] <= now:
            del access_tokens[token]
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        return TokenPayload(**record["user"])


def _authenticate_user(username: str, password: str) -> Optional[TokenPayload]:
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return TokenPayload(id=0, username=username, role=ADMIN_ROLE, display_name=ADMIN_DISPLAY_NAME)
    return None


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> TokenPayload:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    return await _validate_access_token(credentials.credentials)


async def require_admin(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return user


def record_event_log(
    *,
    session_id: Optional[str],
    user_id: Optional[int],
    username: Optional[str],
    operator: Optional[str],
    event_type: str,
    category: Optional[str],
    authorized: Optional[bool],
    payload: Optional[Dict[str, Any]],
    timestamp: Optional[datetime] = None,
) -> None:
    ts = timestamp or _now()
    entry = EventLog(
        session_id=session_id,
        user_id=user_id,
        username=username,
        operator=operator,
        type=event_type,
        category=category,
        authorized=authorized if authorized is not None else True,
        payload=json.dumps(payload, ensure_ascii=False) if payload is not None else None,
        timestamp=ts,
    )
    with SessionLocal() as db:
        db.add(entry)
        db.commit()


def serialize_event_log(log: EventLog) -> LogEntryResponse:
    payload: Optional[Dict[str, Any]] = None
    if log.payload:
        try:
            payload = json.loads(log.payload)
        except json.JSONDecodeError:
            payload = None
    timestamp = log.timestamp or _now()
    return LogEntryResponse(
        id=log.id,
        timestamp=_to_iso(timestamp),
        type=log.type,
        session_id=log.session_id,
        operator=log.operator,
        authorized=log.authorized,
        payload=payload,
        username=log.username,
        user_id=log.user_id,
        category=log.category,
    )
class SpeakerEmbedder:
    def __init__(
        self, 
        model_path="./models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx", 
        sample_rate=16000, 
        threshold=0.6
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.model_path,
            num_threads=4,
            provider="cpu"
        )
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(self.config)
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
        self.threshold = threshold
        self.sample_rate = sample_rate

    def create_stream(self):
        return self.extractor.create_stream()

    def is_ready(self, stream):
        return self.extractor.is_ready(stream)

    def compute(self, stream):
        return self.extractor.compute(stream)

    def embed(self, samples, sample_rate):
        stream = self.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        if not self.is_ready(stream):
            raise RuntimeError("Speaker embedding extractor is not ready")

        embedding = self.compute(stream)
        return np.asarray(embedding, dtype=np.float32)


def _pcm_bytes_to_float32(data: bytes, dtype: Optional[str]) -> np.ndarray:
    try:
        x = np.frombuffer(data, dtype=np.float32)
        if x.size == 0:
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return x
    except Exception:
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

def _ms(samples: int, sr: int) -> int:
    return int(samples * 1000 / sr)

class AsrSession:
    def __init__(self, websocket: WebSocket, app: FastAPI):
        self.ws = websocket
        self.app = app
        self.args = app.state.args
        self.recognizer = app.state.recognizer
        self.embedder: SpeakerEmbedder = app.state.embedder

        self.sample_rate_client = getattr(self.embedder, "sample_rate", None) or self.args.sample_rate or 16000
        self.dtype = "float32"
        self.dtype_hint = "float32"

        self.stream = self.recognizer.create_stream()
        self.total_samples_in = 0
        self.cur_utt_start_sample = 0
        self.cur_utt_audio = []
        self.cur_utt_speaker_guess_sent = False
        self.segment_id = 0

    def _concat_cur_utt_audio(self) -> np.ndarray:
        if not self.cur_utt_audio:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.cur_utt_audio, axis=0)

    def _try_speaker(self, force: bool = False) -> Tuple[str, float]:
        buf = self._concat_cur_utt_audio()
        need_len = int(self.args.min_spk_seconds * self.sample_rate_client)
        if (not force) and (buf.size < need_len):
            return "unknown", 0.0

        st = self.embedder.create_stream()
        st.accept_waveform(sample_rate=self.sample_rate_client, waveform=buf)
        if force:
            st.input_finished()
        if not self.embedder.is_ready(st):
            return "unknown", 0.0
        emb = self.embedder.compute(st)
        emb = np.asarray(emb, dtype=np.float32)

        matched, top_sim, _topk = identify_user(emb, threshold=self.args.threshold)

        if force:
            metrics = getattr(self.app.state, "session_metrics", None)
            if metrics is not None:
                sims = metrics.setdefault("similarity_samples", [])
                sims.append(float(top_sim))
                if len(sims) > 200:
                    metrics["similarity_samples"] = sims[-200:]
                if matched is None:
                    metrics["undetermined_count"] = int(metrics.get("undetermined_count", 0)) + 1

        if matched is None:
            return "unknown", float(top_sim)
        return matched, float(top_sim)

    async def _send_partial(self, text: str, speaker: str):
        await self.ws.send_json({
            "type": "partial",
            "segment_id": self.segment_id,
            "start_ms": _ms(self.cur_utt_start_sample, self.sample_rate_client),
            "time_ms": _ms(self.total_samples_in, self.sample_rate_client),
            "text": text,
            "speaker": speaker,
        })

    async def _send_final(self, text: str, speaker: str, similarity: float):
        await self.ws.send_json({
            "type": "final",
            "segment_id": self.segment_id,
            "start_ms": _ms(self.cur_utt_start_sample, self.sample_rate_client),
            "end_ms": _ms(self.total_samples_in, self.sample_rate_client),
            "text": text,
            "speaker": speaker,
            "similarity": similarity,
        })

        record_event_log(
            session_id=self.ws.scope.get("session_id"),
            user_id=None,
            username=speaker if speaker != "unknown" else None,
            operator=speaker if speaker != "unknown" else None,
            event_type="transcript",
            category="final",
            authorized=speaker != "unknown",
            payload={
                "text": text,
                "segment_id": self.segment_id,
                "similarity": similarity,
                "start_ms": _ms(self.cur_utt_start_sample, self.sample_rate_client),
                "end_ms": _ms(self.total_samples_in, self.sample_rate_client),
            },
        )

        self.segment_id += 1
        self.cur_utt_start_sample = self.total_samples_in
        self.cur_utt_audio.clear()
        self.cur_utt_speaker_guess_sent = False

    async def handle_binary_audio(self, data: bytes):
        # 1) 入队解析
        samples = _pcm_bytes_to_float32(data, self.dtype_hint)
        self.total_samples_in += samples.size
        self.cur_utt_audio.append(samples)
        metrics = getattr(self.app.state, "session_metrics", None)
        if metrics is not None:
            metrics["audio_queue_depth"] = max(
                int(metrics.get("audio_queue_depth", 0)),
                len(self.cur_utt_audio),
            )

        # 2) 喂给 ASR（内部重采样）
        self.stream.accept_waveform(self.sample_rate_client, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        # 3) 拿到当前文本 + 可能的说话人猜测
        text = self.recognizer.get_result(self.stream)
        speaker = "unknown"
        if not self.cur_utt_speaker_guess_sent:
            guess, _sim = self._try_speaker(force=False)
            speaker = guess
            if guess != "unknown":
                self.cur_utt_speaker_guess_sent = True

        await self._send_partial(text, speaker)

        # 4) 端点：定格该话段并重置
        if self.recognizer.is_endpoint(self.stream):
            final_spk, final_sim = self._try_speaker(force=True)
            await self._send_final(text, final_spk, final_sim)
            self.recognizer.reset(self.stream)

    async def handle_done(self):
        """
        客户端结束本次识别：
        - 若当前话段已有文本，则强制做一次最终说话人识别并发送 final
        - 然后由上层发送 {"type":"done"} 并关闭连接
        """
        text = self.recognizer.get_result(self.stream)
        if text.strip():
            final_spk, final_sim = self._try_speaker(force=True)
            await self._send_final(text, final_spk, final_sim)

# Utility functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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

    args = parser.parse_args()

    return args

def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def identify_user(query_embedding: np.ndarray, threshold: float) -> tuple:
    sims = []
    if query_embedding is None or query_embedding.size == 0:
        return None, 0.0, sims

    with SessionLocal() as db:
        users = db.query(User).filter(User.embedding.isnot(None)).all()

        for user in users:
            if not user.embedding:
                continue
            try:
                stored_embedding = np.array(json.loads(user.embedding), dtype=np.float32)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if stored_embedding.size == 0:
                continue
            sim = cosine_similarity(query_embedding, stored_embedding)
            sims.append((user.username, sim))

    if not sims:
        return None, 0.0, []

    sims.sort(key=lambda x: x[1], reverse=True)
    topk = sims[:5]
    top_user, top_sim = topk[0]

    matched = top_user if top_sim >= threshold else None
    return matched, float(top_sim), topk

def create_recognizer(
    tokens: str = "",
    encoder: str = "",
    decoder: str = "",
    joiner: str = "",
    num_threads: int = 1,
    sample_rate: int = 16000,
    feature_dim: int = 80,
    decoding_method: str = "greedy_search",
    max_active_paths: int = 4,
    provider: str = "cpu",
    hotwords_file: str = "",
    hotwords_score: int = 1.5,
    blank_penalty: float = 0.0,
    hr_rule_fsts: str = "",
    hr_lexicon: str = "",    
):
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=num_threads,
        sample_rate=sample_rate,
        feature_dim=feature_dim,
        decoding_method=decoding_method,
        max_active_paths=max_active_paths,
        provider=provider,
        hotwords_file=hotwords_file,
        hotwords_score=hotwords_score,
        blank_penalty=blank_penalty,
        hr_rule_fsts=hr_rule_fsts,
        hr_lexicon=hr_lexicon,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
    )

    return recognizer

def create_app(args):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.args = args
        app.state.start_time = _now()
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
            threshold=args.threshold
        )
        yield

    app = FastAPI(lifespan=lifespan)

    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/auth/login", response_model=TokenResponse, status_code=status.HTTP_200_OK)
    async def login(payload: LoginRequest):
        user_payload = _authenticate_user(payload.username, payload.password)
        if not user_payload:
            record_event_log(
                session_id=None,
                user_id=None,
                username=payload.username,
                operator=None,
                event_type="auth",
                category="login_failed",
                authorized=False,
                payload={"reason": "INVALID_CREDENTIALS"},
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")

        token_response = await _issue_tokens(user_payload)
        record_event_log(
            session_id=None,
            user_id=user_payload.id,
            username=user_payload.username,
            operator=user_payload.display_name,
            event_type="auth",
            category="login",
            authorized=True,
            payload={"expires_in": token_response.expires_in},
        )
        return token_response

    @app.post("/auth/refresh", response_model=RefreshResponse, status_code=status.HTTP_200_OK)
    async def refresh_token(payload: RefreshRequest):
        token_response = await _refresh_access_token(payload.refresh_token)
        return RefreshResponse(token=token_response.token, expires_in=token_response.expires_in, refresh_token=token_response.refresh_token)

    # User相关
    @app.get("/users", response_model=list[UserResponse])
    async def get_users(
        db: Session = Depends(get_db),
        current_user: TokenPayload = Depends(get_current_user),
    ):
        users = db.query(User).all()
        return [
            UserResponse(
                id=u.id,
                username=u.username,
                identity=u.identity,
                has_voiceprint=bool(u.embedding)
            )
            for u in users
        ]

    @app.get("/users/{user_id}", response_model=UserResponse)
    async def get_user_by_id(
        user_id: int,
        db: Session = Depends(get_db),
        current_user: TokenPayload = Depends(get_current_user),
    ):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            id=user.id, username=user.username, identity=user.identity, has_voiceprint=bool(user.embedding)
        )
    
    @app.post("/users", response_model=UserResponse, status_code=201)
    async def create_user(
        payload: UserCreateAndUpdate,
        db: Session = Depends(get_db),
        _: TokenPayload = Depends(require_admin),
    ):
        user = User(username=payload.username, identity=payload.identity)
        db.add(user)
        db.commit()
        db.refresh(user)
        return UserResponse(
            id=user.id,
            username=user.username,
            identity=user.identity,
            has_voiceprint=bool(getattr(user, "embedding", None)),
        )

    @app.patch("/users/{user_id}", response_model=UserResponse)
    async def update_user_by_id(
        user_id: int,
        payload: UserCreateAndUpdate,
        db: Session = Depends(get_db),
        _: TokenPayload = Depends(require_admin),
    ):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if payload.username and payload.username != user.username:
            exists = db.query(User).filter(User.username == payload.username).first()
            if exists:
                raise HTTPException(status_code=409, detail="Username already exists")
            user.username = payload.username

        if payload.identity is not None:
            user.identity = payload.identity

        db.commit()
        db.refresh(user)
        return UserResponse(
            id=user.id, username=user.username, identity=user.identity, has_voiceprint=bool(user.embedding)
        )

    @app.delete("/users/{user_id}", status_code=204)
    async def delete_user_by_id(
        user_id: int,
        db: Session = Depends(get_db),
        _: TokenPayload = Depends(require_admin),
    ):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        db.delete(user)
        db.commit()
        return Response(status_code=204)

    @app.delete("/users/by-username/{username}", status_code=204)
    async def delete_user_by_username(
        username: str,
        db: Session = Depends(get_db),
        _: TokenPayload = Depends(require_admin),
    ):
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        db.delete(user)
        db.commit()
        return Response(status_code=204)
    
    # 识别相关
    @app.post("/users/{user_id}/voiceprint/aggregate", response_model=UserResponse)
    async def embedding(
        user_id: int,
        files: List[UploadFile],
        db: Session = Depends(get_db),
        _: TokenPayload = Depends(require_admin),
    ):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if len(files) != 3:
            raise HTTPException(status_code=400, detail=f"Need 3 sound files, but actual {len(files)}")
        
        embedder = app.state.embedder
        ans = None
        for f in files:
            raw = await f.read()
            if not raw:
                raise HTTPException(status_code=400, detail=f"File {f.filename} is NULL")

            try:
                with io.BytesIO(raw) as bio:
                    data, sample_rate = sf.read(bio, always_2d=True, dtype="float32")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode {f.filename}: {e}") from e

            if sample_rate != embedder.sample_rate:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample rate mismatch for {f.filename}: expected {embedder.sample_rate}, got {sample_rate}"
                )

            data = data[:, 0]  # use only the first channel
            samples = np.ascontiguousarray(data, dtype=np.float32)
            embedding = embedder.embed(samples, sample_rate)

            if ans is None:
                ans = embedding
            else:
                ans += embedding

        if ans is None:
            raise HTTPException(status_code=500, detail="Failed to compute embeddings")

        ans = ans / len(files)
        user.embedding = json.dumps(ans.tolist())

        db.commit()
        db.refresh(user)

        return UserResponse(
            id=user.id,
            username=user.username,
            identity=user.identity,
            has_voiceprint=True
        )

    @app.get("/logs", response_model=LogsResponse)
    async def get_logs(
        type: Optional[str] = Query(None, description="Filter by log type"),
        authorized: Optional[bool] = Query(None),
        user_id: Optional[int] = Query(None),
        username: Optional[str] = Query(None),
        from_ts: Optional[str] = Query(None, alias="from"),
        to_ts: Optional[str] = Query(None, alias="to"),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
        current_user: TokenPayload = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        query = db.query(EventLog)

        if type:
            query = query.filter(EventLog.type == type)
        if authorized is not None:
            query = query.filter(EventLog.authorized == authorized)
        if user_id is not None:
            query = query.filter(EventLog.user_id == user_id)
        if username is not None:
            query = query.filter(EventLog.username == username)

        from_dt = _parse_iso8601(from_ts)
        to_dt = _parse_iso8601(to_ts)
        if from_dt:
            query = query.filter(EventLog.timestamp >= from_dt)
        if to_dt:
            query = query.filter(EventLog.timestamp <= to_dt)

        total = query.with_entities(func.count(EventLog.id)).scalar() or 0
        items = (
            query.order_by(EventLog.timestamp.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        data = [serialize_event_log(item) for item in items]
        return LogsResponse(items=data, total=total, page=page, page_size=page_size)

    @app.get("/status/health", response_model=HealthResponse)
    async def health():
        start_time: Optional[datetime] = getattr(app.state, "start_time", None)
        uptime = int((_now() - start_time).total_seconds()) if start_time else 0
        asr_ready = getattr(app.state, "recognizer", None) is not None
        speaker_ready = getattr(app.state, "embedder", None) is not None
        db_status = "ok"
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            db_status = "error"

        args = getattr(app.state, "args", argparse.Namespace())
        asr_model = getattr(args, "encoder", None)
        speaker_model = getattr(args, "model_path", None)

        return HealthResponse(
            status="ok" if all([asr_ready, speaker_ready, db_status == "ok"]) else "degraded",
            uptime_seconds=uptime,
            asr_model=asr_model,
            speaker_model=speaker_model,
            asr_ready=asr_ready,
            speaker_ready=speaker_ready,
            database=db_status,
        )

    @app.get("/status/metrics", response_model=MetricsResponse)
    async def metrics(_: TokenPayload = Depends(require_admin)):
        metrics = getattr(app.state, "session_metrics", None) or {}
        latency_samples: List[float] = metrics.get("latency_samples", [])
        similarity_samples: List[float] = metrics.get("similarity_samples", [])
        undetermined = metrics.get("undetermined_count", 0)
        audio_depth = metrics.get("audio_queue_depth", 0)

        avg_latency = float(sum(latency_samples) / len(latency_samples)) if latency_samples else 0.0
        avg_similarity = float(sum(similarity_samples) / len(similarity_samples)) if similarity_samples else 0.0
        active_sessions = int(metrics.get("active_sessions", 0))
        undetermined_rate = (
            float(undetermined / len(similarity_samples)) if similarity_samples else 0.0
        )

        return MetricsResponse(
            active_sessions=active_sessions,
            avg_recognition_latency_ms=avg_latency,
            avg_embedding_similarity=avg_similarity,
            undetermined_speaker_rate=undetermined_rate,
            audio_queue_depth=int(audio_depth),
        )

    @app.websocket("/ws/asr")
    async def ws_identify(websocket: WebSocket):
        session_id = f"sess-{int(_now().timestamp() * 1000)}-{random.randint(1000, 9999)}"
        await websocket.accept()
        websocket.scope["session_id"] = session_id

        metrics = getattr(app.state, "session_metrics", None)
        if metrics is not None:
            metrics["active_sessions"] = int(metrics.get("active_sessions", 0)) + 1

        record_event_log(
            session_id=session_id,
            user_id=None,
            username=None,
            operator=None,
            event_type="session",
            category="open",
            authorized=True,
            payload={"session_id": session_id},
        )

        session = AsrSession(websocket, app)

        try:
            while True:
                msg = await websocket.receive()

                if msg.get("bytes") is not None:
                    await session.handle_binary_audio(msg["bytes"])
                    continue

                txt = msg.get("text")
                if txt is not None:
                    if txt.strip().upper() == "DONE":
                        await session.handle_done()
                        await websocket.send_json({"type": "done"})
                        await websocket.close()
                        return

        except WebSocketDisconnect:
            record_event_log(
                session_id=session_id,
                user_id=None,
                username=None,
                operator=None,
                event_type="session",
                category="disconnect",
                authorized=True,
                payload={"reason": "client_disconnected"},
            )
            return
        except Exception as e:
            record_event_log(
                session_id=session_id,
                user_id=None,
                username=None,
                operator=None,
                event_type="session",
                category="error",
                authorized=False,
                payload={"error": str(e)},
            )
            try:
                await websocket.send_json({"type": "error", "msg": str(e)})
            except Exception:
                pass
            raise
        finally:
            if metrics is not None:
                metrics["active_sessions"] = max(0, int(metrics.get("active_sessions", 1)) - 1)
            record_event_log(
                session_id=session_id,
                user_id=None,
                username=None,
                operator=None,
                event_type="session",
                category="close",
                authorized=True,
                payload={"session_id": session_id},
            )

    return app

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
