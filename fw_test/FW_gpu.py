import io
import os
import queue
import threading
import time
import uuid
import asyncio
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16000
MODEL_SIZE = "small"
COMPUTE_TYPE = os.getenv("FW_COMPUTE_TYPE", "float16")
CUDA_DEVICES = os.getenv("CUDA_DEVICES", "0")
WORKERS_PER_DEVICE = int(os.getenv("FW_WORKERS_PER_DEVICE", "4"))
TOTAL_WORKERS = int(os.getenv("FW_TOTAL_WORKERS", "0"))
QUEUE_MAXSIZE = int(os.getenv("FW_QUEUE_MAXSIZE", "128"))

VAD_THRESHOLD = float(os.getenv("FW_VAD_THRESHOLD", "0.45"))
VAD_MIN_SPEECH_MS = int(os.getenv("FW_VAD_MIN_SPEECH_MS", "80"))
VAD_MIN_SILENCE_MS = int(os.getenv("FW_VAD_MIN_SILENCE_MS", "500"))
MIN_AUDIO_SEC = float(os.getenv("FW_MIN_AUDIO_SEC", "0.20"))
PARTIAL_INTERVAL_SEC = float(os.getenv("FW_PARTIAL_INTERVAL_SEC", "0.8"))
MAX_SEGMENT_SEC = float(os.getenv("FW_MAX_SEGMENT_SEC", "2.0"))


def parse_cuda_devices(raw: str) -> list[int]:
    devices = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        devices.append(int(token))
    if not devices:
        raise RuntimeError("No CUDA device is configured. Set CUDA_DEVICES.")
    return devices


def build_worker_device_plan(device_indices: list[int]) -> list[int]:
    if TOTAL_WORKERS > 0:
        return [device_indices[i % len(device_indices)] for i in range(TOTAL_WORKERS)]
    if WORKERS_PER_DEVICE < 1:
        raise RuntimeError("FW_WORKERS_PER_DEVICE must be >= 1.")
    plan = []
    for device_index in device_indices:
        plan.extend([device_index] * WORKERS_PER_DEVICE)
    return plan


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype("float32")
    return np.mean(audio, axis=1).astype("float32")


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype("float32")
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError("Invalid sample rate.")
    src_len = len(audio)
    if src_len == 0:
        return np.array([], dtype="float32")
    dst_len = int(round(src_len * dst_sr / src_sr))
    if dst_len <= 1:
        return np.array([], dtype="float32")
    src_x = np.arange(src_len, dtype=np.float64)
    dst_x = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float64)
    return np.interp(dst_x, src_x, audio).astype("float32")


def preprocess_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    mono = to_mono(audio)
    out = resample_linear(mono, sr, SAMPLE_RATE)
    if len(out) == 0:
        return out

    # Remove DC offset and apply mild peak normalization.
    out = out - np.mean(out)
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = np.clip(out / peak * 0.95, -1.0, 1.0)
    return out.astype("float32")


def preprocess_pcm16_mono(chunk: bytes) -> np.ndarray:
    if not chunk:
        return np.array([], dtype="float32")
    pcm = np.frombuffer(chunk, dtype=np.int16)
    if len(pcm) == 0:
        return np.array([], dtype="float32")
    return (pcm.astype("float32") / 32768.0).clip(-1.0, 1.0)


@dataclass
class Job:
    job_id: str
    created_at: float
    language: str
    audio: np.ndarray
    loop: Any
    future: Any
    enqueue_seq: int


app = FastAPI(title="FW GPU STT", version="1.0.0")
job_queue: queue.Queue[Job] = queue.Queue(maxsize=QUEUE_MAXSIZE)
worker_status: dict[int, str] = {}
workers_started = False
enqueue_counter = itertools.count(1)
dequeue_counter = itertools.count(1)


def run_transcribe(audio: np.ndarray, language: str, model: WhisperModel, vad_model: Any) -> dict[str, Any]:
    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
    )
    if not speech_timestamps:
        return {
            "text": "",
            "segments": [],
            "speech_count": 0,
            "speech_sec": 0.0,
        }

    speech_audio = np.concatenate([audio[s["start"]:s["end"]] for s in speech_timestamps]).astype("float32")
    speech_sec = len(speech_audio) / SAMPLE_RATE
    if speech_sec < MIN_AUDIO_SEC:
        return {
            "text": "",
            "segments": [],
            "speech_count": len(speech_timestamps),
            "speech_sec": speech_sec,
        }

    segments, info = model.transcribe(
        speech_audio,
        language=language,
        vad_filter=False,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    items = []
    texts = []
    for seg in segments:
        text = seg.text.strip()
        items.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            }
        )
        if text:
            texts.append(text)

    return {
        "text": " ".join(texts).strip(),
        "segments": items,
        "speech_count": len(speech_timestamps),
        "speech_sec": speech_sec,
        "language": getattr(info, "language", language),
    }


def worker_loop(worker_id: int, device_index: int) -> None:
    print(f"[worker-{worker_id}] loading model on cuda:{device_index}", flush=True)
    model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        device_index=device_index,
        compute_type=COMPUTE_TYPE,
    )
    vad_model = load_silero_vad()
    print(f"[worker-{worker_id}] ready on cuda:{device_index}", flush=True)

    while True:
        job = job_queue.get()
        worker_status[worker_id] = f"busy:{job.job_id}"
        dequeue_seq = next(dequeue_counter)
        try:
            result = run_transcribe(job.audio, job.language, model, vad_model)
            payload = {
                "job_id": job.job_id,
                "enqueue_seq": job.enqueue_seq,
                "dequeue_seq": dequeue_seq,
                "worker_id": worker_id,
                "device_index": device_index,
                "queue_wait_sec": max(0.0, time.time() - job.created_at),
                **result,
            }
            job.loop.call_soon_threadsafe(job.future.set_result, payload)
        except Exception as e:
            job.loop.call_soon_threadsafe(job.future.set_exception, e)
        finally:
            worker_status[worker_id] = "idle"
            job_queue.task_done()


def start_workers_once() -> None:
    global workers_started
    if workers_started:
        return
    device_indices = parse_cuda_devices(CUDA_DEVICES)
    worker_plan = build_worker_device_plan(device_indices)
    for worker_id, device_index in enumerate(worker_plan):
        worker_status[worker_id] = "starting"
        t = threading.Thread(
            target=worker_loop,
            args=(worker_id, device_index),
            daemon=True,
            name=f"fw-gpu-worker-{worker_id}",
        )
        t.start()
    workers_started = True


async def submit_job(processed: np.ndarray, language: str) -> dict[str, Any]:
    if job_queue.full():
        raise HTTPException(status_code=429, detail="Queue is full. Try again later.")

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    job = Job(
        job_id=str(uuid.uuid4()),
        created_at=time.time(),
        language=language,
        audio=processed,
        loop=loop,
        future=future,
        enqueue_seq=next(enqueue_counter),
    )
    # queue.Queue is FIFO, so earliest request gets the next free worker.
    job_queue.put_nowait(job)
    return await future


@app.on_event("startup")
def on_startup() -> None:
    start_workers_once()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "cuda_devices": CUDA_DEVICES,
        "workers_per_device": WORKERS_PER_DEVICE,
        "total_workers": len(worker_status),
        "workers": worker_status,
        "queue_size": job_queue.qsize(),
        "queue_maxsize": QUEUE_MAXSIZE,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("ko"),
) -> dict[str, Any]:
    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}") from e

    processed = preprocess_audio(np.asarray(audio), int(sr))
    if len(processed) == 0:
        raise HTTPException(status_code=400, detail="Audio has no usable samples.")

    return await submit_job(processed, language)


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "status": "ready",
            "type": "info",
            "sample_rate": SAMPLE_RATE,
            "format": "pcm_s16le_mono",
        }
    )

    language = "ko"
    buffer: list[np.ndarray] = []
    last_partial_at = time.time()

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_data = message.get("text")
            if text_data:
                cmd = text_data.strip().lower()
                if cmd.startswith("lang:"):
                    language = cmd.split(":", 1)[1].strip() or "ko"
                    await websocket.send_json({"status": "ok", "type": "config", "language": language})
                    continue
                if cmd in ("flush", "final"):
                    if not buffer:
                        await websocket.send_json({"status": "listening", "type": "silence", "text": ""})
                        continue
                    audio = np.concatenate(buffer).astype("float32")
                    buffer.clear()
                    result = await submit_job(audio, language)
                    await websocket.send_json(
                        {
                            "status": "completed",
                            "type": "final",
                            "text": result.get("text", ""),
                            "segments": result.get("segments", []),
                            "worker_id": result.get("worker_id"),
                            "device_index": result.get("device_index"),
                        }
                    )
                    last_partial_at = time.time()
                    continue
                if cmd in ("stop", "close"):
                    await websocket.close()
                    break

            bytes_data = message.get("bytes")
            if bytes_data:
                chunk = preprocess_pcm16_mono(bytes_data)
                if len(chunk) == 0:
                    continue
                buffer.append(chunk)

                buffered_sec = sum(len(x) for x in buffer) / SAMPLE_RATE
                now = time.time()
                should_partial = buffered_sec >= MIN_AUDIO_SEC and (now - last_partial_at) >= PARTIAL_INTERVAL_SEC
                should_force = buffered_sec >= MAX_SEGMENT_SEC
                if not (should_partial or should_force):
                    continue

                audio = np.concatenate(buffer).astype("float32")
                buffer.clear()
                result = await submit_job(audio, language)
                text = result.get("text", "")
                if text:
                    await websocket.send_json(
                        {
                            "status": "streaming",
                            "type": "partial",
                            "text": text,
                            "segments": result.get("segments", []),
                            "worker_id": result.get("worker_id"),
                            "device_index": result.get("device_index"),
                        }
                    )
                else:
                    await websocket.send_json({"status": "listening", "type": "silence", "text": ""})
                last_partial_at = now
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("FW_gpu:app", host="0.0.0.0", port=8000, reload=False)
