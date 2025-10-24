from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from typing import Optional
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

# SQLAlchemy Database Setup
DATABASE_URL = "sqlite:///./voiceprints.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    embedding = Column(Text, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    dim = Column(Integer, nullable=False)


Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    username: str

class UserResponse(BaseModel):
    id: int
    username: str
    dim: int
    sample_rate: int


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
        self.threshold = threshold

    def embed(self, samples, sample_rate):

        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        assert self.extractor.is_ready(stream)
        embedding = self.extractor.compute(stream)
        embedding = np.array(embedding)

        return embedding
    
class SpeechRecognizer:
    def __init__(
        self, 
        tokens="",
        encoder="",
        decoder="",
        joiner="",
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        max_active_paths=4,
        provider="cpu",
        hotwords_file="",
        hotwords_score=1.5,
        blank_penalty=0.0,
        hr_rule_fsts="",
        hr_lexicon="",
    ):
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
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
        )

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

def store_user_embedding(username, embedding, sr):
    db = next(get_db())
    embedding_json = json.dumps(embedding.tolist())
    dim = len(embedding)

    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        existing_user.embedding = embedding_json
        existing_user.sample_rate = sr
        existing_user.dim = dim
        db.commit()
        return existing_user.id

    new_user = User(username=username, embedding=embedding_json, sample_rate=sr, dim=dim)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user.id

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


def create_app(args):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.args = args

        app.state.recognizer = SpeechRecognizer(
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

    @app.websocket("/ws/register/{username}")
    async def ws_register(websocket: WebSocket, username: str):
        await websocket.accept()
        embedder = websocket.app.state.embedder
        recognizer = websocket.app.state.recognizer

        raw_buf = bytearray()
        stream = recognizer.recognizer.create_stream()

        client_sample_rate = 16000

        text = ""

        try:
            while True:
                msg = await websocket.receive()

                if "bytes" in msg and msg["bytes"]:
                    b = msg["bytes"]
                    raw_buf.extend(b)
                    pcm = np.frombuffer(b, dtype=np.float32)

                    stream.accept_waveform(sample_rate=client_sample_rate, waveform=pcm)

                    while recognizer.recognizer.is_ready(stream):
                        recognizer.recognizer.decode_stream(stream)

                    r = recognizer.get_result(stream)
                    text = r.text if hasattr(r, "text") else str(r)
                    if text != last_text:
                        last_text = text
                        await websocket.send_json({"final": False, "asr": {"text": text}})

                elif "text" in msg and msg["text"]:
                    ctrl = msg["text"].strip().upper()
                    if ctrl == "DONE":

                        stream.input_finished()
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)
                        r = recognizer.get_result(stream)
                        final_text = r.text if hasattr(r, "text") else str(r)

                        wav = np.frombuffer(bytes(raw_buf), dtype=np.float32)

                        query_embedding = await asyncio.to_thread(embedder.embed, wav, int(client_sample_rate))

                        matched, sim, topk = await asyncio.to_thread(identify_user, query_embedding, embedder.threshold)

                        await websocket.send_json({
                            "final": True,
                            "asr": {"text": final_text},
                            "speaker": {
                                "matched": matched,
                                "similarity": sim,
                                "topk": topk,
                                "threshold": embedder.threshold,
                            }
                        })
                        break
                    else:
                        await websocket.send_json({"notice": f"Unknown control: {ctrl}"})

        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()

    @app.websocket("/ws/asr")
    async def asr(websocket):
        pass

    def decode_wav_from_bytes(raw: bytes):
        with io.BytesIO(raw) as bio:
            data, sr = sf.read(bio, dtype="float32", always_2d=True)
        return np.mean(data, axis=1), int(sr)

    @app.websocket("/ws/identify")
    async def ws_identify(websocket: WebSocket, threshold: float = Query(0.6)):
        await websocket.accept()
        embedder = SpeakerEmbedder()
        recognizer = SpeechRecognizer()
        raw_buf = bytearray()
        stream = recognizer.recognizer.create_stream()

        client_sample_rate = 16000

        text = ""

        try:
            while True:
                msg = await websocket.receive()

                if "bytes" in msg and msg["bytes"]:
                    b = msg["bytes"]
                    raw_buf.extend(b)
                    pcm = np.frombuffer(b, dtype=np.float32)

                    stream.accept_waveform(sample_rate=client_sample_rate, waveform=pcm)

                    while recognizer.recognizer.is_ready(stream):
                        recognizer.recognizer.decode_stream(stream)

                    r = recognizer.get_result(stream)
                    text = r.text if hasattr(r, "text") else str(r)
                    if text != last_text:
                        last_text = text
                        await websocket.send_json({"final": False, "asr": {"text": text}})

                elif "text" in msg and msg["text"]:
                    ctrl = msg["text"].strip().upper()
                    if ctrl == "DONE":

                        stream.input_finished()
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)
                        r = recognizer.get_result(stream)
                        final_text = r.text if hasattr(r, "text") else str(r)

                        wav = np.frombuffer(bytes(raw_buf), dtype=np.float32)

                        query_embedding = await asyncio.to_thread(embedder.embed, wav, int(client_sample_rate))

                        matched, sim, topk = await asyncio.to_thread(identify_user, query_embedding, threshold)

                        await websocket.send_json({
                            "final": True,
                            "asr": {"text": final_text},
                            "speaker": {
                                "matched": matched,
                                "similarity": sim,
                                "topk": topk,
                                "threshold": threshold,
                            }
                        })
                        break
                    else:
                        await websocket.send_json({"notice": f"Unknown control: {ctrl}"})

        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()

    @app.get("/users/{user_id}", response_model=UserResponse)
    async def get_user(user_id: int, db: Session = Depends(get_db)):
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    
    @app.get("/healthy")
    async def tmp():
        return "health"

    return app

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)