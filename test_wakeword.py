import sounddevice as sd
import numpy as np
from openwakeword.model import Model
import openwakeword

# Download models if not already present
openwakeword.utils.download_models()

print("Loading wake word model...")
model = Model(wakeword_models=["hey_jarvis"])

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # openwakeword expects 80ms chunks at 16kHz

print("\nListening for 'Hey Jarvis'... (Ctrl+C to stop)\n")

def callback(indata, frames, time, status):
    audio = (indata[:, 0] * 32767).astype(np.int16)
    prediction = model.predict(audio)
    for mdl_name, score in prediction.items():
        if score > 0.5:
            print(f"Detected '{mdl_name}'! (confidence: {score:.2f})")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                     blocksize=CHUNK_SIZE, callback=callback):
    while True:
        sd.sleep(100)
