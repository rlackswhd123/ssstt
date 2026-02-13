import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="int8"
)

SAMPLE_RATE = 16000
BUFFER = []

def callback(indata, frames, time, status):
    global BUFFER
    audio = indata[:, 0].astype("float32")
    BUFFER.append(audio)

    # 1초마다 부분 STT
    if len(BUFFER) * frames >= SAMPLE_RATE:
        data = np.concatenate(BUFFER)
        BUFFER.clear()

        segments, _ = model.transcribe(
            data,
            language="ko",
            vad_filter=True
        )

        for s in segments:
            print(s.text)

print("🎤 말해보세요... (Ctrl+C 종료)")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=callback
):
    while True:
        sd.sleep(1000)

