from fastapi import FastAPI, WebSocket, WebSocketDisconnect, \
    HTTPException, Query, Depends, UploadFile, File, Response
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager
import numpy as np
import json
import io
import asyncio
import soundfile as sf
import random
import argparse

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

    def embed(self, samples, sample_rate):

        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        assert self.extractor.is_ready(stream)
        embedding = self.extractor.compute(stream)
        embedding = np.array(embedding)

        return embedding    

    def search(self):
        pass
    
    def __call__(self, *args, **kwds):
        pass


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

        self.sample_rate_client = 16000
        self.dtype = "float32"

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
        emb = self.embedder.compute(st)

        matched, top_sim, _topk = identify_user(emb, threshold=self.args.threshold)
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

    async def _send_final(self, text: str, speaker: str):
        await self.ws.send_json({
            "type": "final",
            "segment_id": self.segment_id,
            "start_ms": _ms(self.cur_utt_start_sample, self.sample_rate_client),
            "end_ms": _ms(self.total_samples_in, self.sample_rate_client),
            "text": text,
            "speaker": speaker,
        })

        self.segment_id += 1
        self.cur_utt_start_sample = self.total_samples_in
        self.cur_utt_audio.clear()
        self.cur_utt_speaker_guess_sent = False

    async def handle_binary_audio(self, data: bytes):
        # 1) 入队解析
        samples = _pcm_bytes_to_float32(data, self.dtype_hint)
        self.total_samples_in += samples.size
        self.cur_utt_audio.append(samples)

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
            final_spk, _sim = self._try_speaker(force=True)
            await self._send_final(text, final_spk)
            self.recognizer.reset(self.stream)

    async def handle_done(self):
        """
        客户端结束本次识别：
        - 若当前话段已有文本，则强制做一次最终说话人识别并发送 final
        - 然后由上层发送 {"type":"done"} 并关闭连接
        """
        text = self.recognizer.get_result(self.stream)
        if text.strip():
            final_spk, _ = self._try_speaker(force=True)
            await self._send_final(text, final_spk)

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

    args = parser.parse_args()

    return args

def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def identify_user(query_embedding: np.ndarray, threshold: float) -> tuple:
    db = next(get_db())
    users = db.query(User).all()

    sims = []
    for user in users:
        stored_embedding = np.array(json.loads(user.embedding), dtype=np.float32)
        sim = cosine_similarity(query_embedding, stored_embedding)
        sims.append((user.username, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    topk = sims[:5]
    top_user, top_sim = sims[0]

    matched = top_user if top_sim >= threshold else None
    return matched, top_sim, topk

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

    # User相关
    @app.get("/users", response_model=list[UserResponse])
    async def get_users(db: Session = Depends(get_db)):
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
    async def get_user_by_id(user_id: int, db: Session = Depends(get_db)):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            id=user.id, username=user.username, identity=user.identity, has_voiceprint=bool(user.embedding)
        )
    
    @app.post("/users", response_model=UserResponse, status_code=201)
    async def create_user(payload: UserCreateAndUpdate, db: Session = Depends(get_db)):
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
    async def update_user_by_id(user_id: int, payload: UserCreateAndUpdate, db: Session = Depends(get_db)):
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
    async def delete_user_by_id(user_id: int, db: Session = Depends(get_db)):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        db.delete(user)
        db.commit()
        return Response(status_code=204)

    @app.delete("/users/by-username/{username}", status_code=204)
    async def delete_user_by_username(username: str, db: Session = Depends(get_db)):
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
        db: Session = Depends(get_db)
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
                raise 

            data = data[:, 0]  # use only the first channel
            samples = np.ascontiguousarray(data)

            stream = embedder.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
            stream.input_finished()

            assert embedder.is_ready(stream)
            embedding = embedder.compute(stream)
            embedding = np.array(embedding)

            if ans is None:
                ans = embedding
            else:
                ans += embedding

        ans = ans / len(files)

        user.embedding = json.dump(ans)
        db.commit()
        db.refresh(user)

        return UserResponse(
            id=user.id,
            username=user.username,
            identity=user.identity,
            has_voiceprint=True
        )

    @app.websocket("/ws/asr")
    async def ws_identify(websocket: WebSocket):
        await websocket.accept()
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
            return
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "msg": str(e)})
            except Exception:
                pass
            raise

    return app

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)